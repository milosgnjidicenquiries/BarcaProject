"""One run of R3P, from a data directory to a validated CSV and its provenance.

    run = pipeline.execute("data", team_name="...")
    run.to_csv("out")            # writes out/<TEAM>.csv
    run.validate()               # the organizers' validator, over the bytes that were written
    run.manifest("out")          # the JSON that makes the claim checkable

The order is the one the rules describe: read, prove what was read, infer, write, validate. Skipping
the integrity step is possible (`skip_checks=True`) but visible -- the manifest records that it was
skipped, so a fast dashboard session cannot be mistaken for a submission run.

`report` carries the internals (emissions, posterior, lattice scores) because the dashboard plots them
and because "everything inspectable" is easier to honour when the numbers a run produced are kept
rather than recomputed later from the CSV. They are not copied into the manifest, which stays a
summary of decisions and parameters.
"""

from __future__ import annotations

import os
import time

import numpy as np

from . import official, settings
from .data import bundle as bundle_module
from .evaluation import scoring
from .possession import decoder, model
from .reporting import manifest, submission, summary

DEFAULT_OUT = os.path.join(official.ROOT, "out")


class Run:
    def __init__(self, cfg, bundle, labels, report, timings, checks_done=True):
        self.cfg = cfg
        self.bundle = bundle
        self.labels = np.asarray(labels, dtype=np.int8)
        self.report = report
        self.timings = timings
        self.checks_done = checks_done
        self.csv_path = None
        self.rows_result = None

    # -- what a run is made of -----------------------------------------------------------------

    @property
    def team_name(self):
        return self.cfg["run"]["team_name"]

    @property
    def n_frames(self):
        return len(self.labels)

    @property
    def graded(self):
        return self.report["arrays"]["graded"]

    def counts(self):
        return {name: int((self.labels == index).sum())
                for index, name in enumerate(summary.LABEL_NAMES)}

    def checks(self):
        return self.bundle.checks

    def failures(self):
        return [row for row in self.bundle.checks if row["status"] == "fail"]

    def warnings(self):
        return [row for row in self.bundle.checks if row["status"] == "warn"]

    # -- artifacts ---------------------------------------------------------------------------

    def to_csv(self, out_dir=DEFAULT_OUT, name=None):
        """Write the submission and keep the hash, so a later run can be compared with this one."""
        path = os.path.join(out_dir, (name or "{}.csv".format(self.team_name)))
        self.rows_result = submission.write_submission(self.labels, path)
        self.csv_path = self.rows_result["path"]
        return self.rows_result

    def validate(self, path=None, n_expected=None):
        """The organizers' validator over the file, plus the two facts that make it mean something.

        `n_expected` defaults to the published video length, which is the contract a real submission is
        held to. A development fixture is shorter, so a test passes its own length and still gets the
        same checks over its bytes -- the difference is stated in the result rather than hidden.
        """
        path = path or self.csv_path
        if path is None:
            raise ValueError("nothing to validate: the run has not written a CSV yet")
        if n_expected is None:
            n_expected = self.cfg["facts"]["n_video_frames"]
        ok, reason = scoring.validate_csv(path, n_frames=n_expected)
        return {"valid": ok, "reason": reason, "path": path, "expected_rows": int(n_expected),
                "rows_written": self.n_frames,
                "rows_match_published_video": self.n_frames == int(n_expected),
                "sha256": submission.sha256_file(path),
                "round_trip": submission.verify_round_trip(self.labels, path)}

    def score(self, gt_path=None, bootstrap_n=None):
        return scoring.score(self.labels, gt_path or scoring.PUBLIC_GT,
                             fps=self.bundle.fps, bootstrap_n=bootstrap_n,
                             n_frames=self.n_frames)

    def compare_with_baselines(self, gt_path=None, bootstrap_n=None):
        return scoring.compare_with_baselines(self.labels, gt_path, bootstrap_n=bootstrap_n)

    def manifest(self, out_dir=DEFAULT_OUT, validator=None, selfcheck=None, extra=None):
        data = manifest.build(self.cfg, self.bundle, self.labels, self.report,
                              csv_path=self.csv_path, validator=validator,
                              selfcheck=selfcheck, timings=self.timings)
        data["integrity_step_skipped"] = not self.checks_done
        if extra:
            data.update(extra)
        path = os.path.join(out_dir, "{}_manifest.json".format(self.team_name))
        manifest.write(data, path)
        transitions = manifest.transitions_csv(data, self.report,
                                               os.path.join(out_dir, "{}_transitions.csv".format(
                                                   self.team_name)))
        uncertainty = manifest.uncertainty_csv(self.report,
                                               os.path.join(out_dir, "{}_uncertainty.csv".format(
                                                   self.team_name)))
        data["artifacts"] = {"manifest": path, "transitions": transitions, "uncertainty": uncertainty}
        manifest.write(data, path)
        return data, path

    # -- dashboard helpers -------------------------------------------------------------------

    def frame_labels(self, start, stop):
        """The decoded labels over a span, for the timeline strip."""
        return np.asarray(self.labels[start:stop], dtype=np.int8)

    def describe(self):
        return {
            "team_name": self.team_name,
            "model_version": self.cfg["meta"]["model_version"],
            "config_digest": settings.config_digest(),
            "data_dir": self.cfg["run"]["data_dir"],
            "source": self.cfg["run"]["source"],
            "n_frames": self.n_frames,
            "graded_frames": self.report["graded_frames"],
            "labels": self.counts(),
            "possessions": self.report["spells"],
            "temperature": self.report["temperature"],
            "development_bundle": bool(self.bundle.development_bundle),
            "checks": {"n": len(self.bundle.checks), "fail": len(self.failures()),
                       "warn": len(self.warnings())},
            "timings": self.timings,
        }


def execute(data_dir=None, team_name=None, source="tactical", cfg=None, out_dir=None,
            progress=None, with_tagging=True, with_homography=True, skip_checks=False,
            game=official.GAME):
    """Read, check, infer. Returns a Run. Writes nothing -- the caller decides what becomes a file."""
    started = time.time()
    cfg = cfg or settings.resolve(settings.load_config(), game=game, data_dir=data_dir,
                                  team_name=team_name, source=source)
    if team_name:
        cfg["run"]["team_name"] = team_name

    def note(done, total, message):
        if progress:
            progress(int(done), int(total), message)

    note(0, 1, "loading the bundle")
    loaded = bundle_module.load_bundle(data_dir=data_dir, cfg=cfg, source=source,
                                       with_tagging=with_tagging,
                                       with_homography=with_homography,
                                       progress=note, game=game)
    read_seconds = round(time.time() - started, 1)

    # The checks themselves run inside load_bundle and cannot be turned off; what skip_checks controls
    # is whether a failure is allowed to stop the run, which the CLI decides and the manifest records.
    labels, report = model.run(loaded, cfg, progress=note)
    return Run(cfg, loaded, labels, report,
               {"bundle_read": read_seconds,
                "total": round(time.time() - started, 1)},
               checks_done=not skip_checks)
