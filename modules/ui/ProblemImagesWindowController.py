"""Reads what dataset curation has decided, and lets the user answer back.

The window runs on the UI thread while training runs on its own, and the two never share objects.
Everything here goes through files in the workspace: the trainer writes its verdicts to
``problem_images.json`` at each epoch boundary, and an edited caption is left in ``caption_edits.json``
for the trainer to pick up at the next one. That keeps a slow disk or a closed window from ever
blocking training.
"""

import json
import os

CURATION_DIR = "loss_log"
REPORT_FILENAME = "problem_images.json"
EDIT_QUEUE_FILENAME = "caption_edits.json"

# Verdicts worth showing, worst first. Anything else is an image doing fine.
VERDICT_ORDER = ["excluded", "stuck", "suspect", "exhausted", "watch", "learning"]

VERDICT_HELP = {
    "excluded": "Dropped from training — its captions were rewritten twice and it stayed stuck.",
    "stuck": "High loss and not coming down. Usually the caption does not match the image.",
    "suspect": "Extreme loss this early is suspicious, but there is no trend to confirm it yet.",
    "exhausted": "Learned what it could, then plateaued. Eased off to avoid overbaking it.",
    "watch": "Looked suspicious this epoch. Not confirmed.",
    "learning": "Hard, but the loss is coming down. This is a good difficult image.",
}


class ProblemImage:
    """One row: what the trainer said about an image, and its caption."""

    def __init__(self, key: str, entry: dict):
        self.key = key
        self.verdict = str(entry.get("verdict", "")).lower()
        self.multiplier = entry.get("multiplier")
        self.mean_residual = entry.get("mean_residual")
        # the per-epoch slope of the image's residual: negative means it is coming down
        self.trend = entry.get("slope")
        self.improving = bool(entry.get("improving"))
        self.epochs = entry.get("epochs")
        self.stuck_epochs = entry.get("stuck_epochs")

    @property
    def name(self) -> str:
        return os.path.basename(self.key)

    @property
    def exists(self) -> bool:
        return os.path.isfile(self.key)

    @property
    def caption_path(self) -> str:
        return os.path.splitext(self.key)[0] + ".txt"

    def read_caption(self) -> str:
        try:
            with open(self.caption_path, encoding="utf-8-sig") as f:
                return f.read().strip()
        except Exception:
            return ""

    def summary(self) -> str:
        parts = [self.verdict or "unknown"]
        if isinstance(self.multiplier, (int, float)) and self.multiplier != 1.0:
            parts.append(f"training at x{self.multiplier:.2f}")
        if isinstance(self.mean_residual, (int, float)):
            parts.append(f"residual {self.mean_residual:+.4f}")
        if isinstance(self.trend, (int, float)):
            direction = "improving" if self.improving else "flat or worsening"
            parts.append(f"{direction} ({self.trend:+.4f}/epoch)")
        if isinstance(self.epochs, int) and self.epochs:
            parts.append(f"{self.epochs} epochs seen")
        if isinstance(self.stuck_epochs, int) and self.stuck_epochs > 1:
            parts.append(f"stuck {self.stuck_epochs} epochs")
        return " · ".join(parts)


class ProblemImagesWindowController:
    def __init__(self, config):
        self.config = config
        self.images: list[ProblemImage] = []
        self.generated_at = None
        self.plateau: dict | None = None
        self._last_mtime = None

    # --- where things live -------------------------------------------------------------------

    @property
    def report_path(self) -> str:
        return os.path.join(self.config.workspace_dir, CURATION_DIR, REPORT_FILENAME)

    @property
    def edit_queue_path(self) -> str:
        return os.path.join(self.config.workspace_dir, CURATION_DIR, EDIT_QUEUE_FILENAME)

    # --- reading -----------------------------------------------------------------------------

    def has_changed(self) -> bool:
        """Whether the trainer has written a new report since the last load."""
        try:
            mtime = os.path.getmtime(self.report_path)
        except OSError:
            return self._last_mtime is not None
        return mtime != self._last_mtime

    def load(self) -> bool:
        """Re-read the report. Returns whether anything is there to show."""
        path = self.report_path
        try:
            self._last_mtime = os.path.getmtime(path)
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception:
            self.images = []
            self.plateau = None
            self._last_mtime = None
            return False

        entries = data.get("images") if isinstance(data, dict) else data
        if not isinstance(entries, dict):
            entries = {}

        self.generated_at = data.get("epoch") if isinstance(data, dict) else None
        self.plateau = {
            "plateaued": bool(data.get("plateaued")),
            "pending": data.get("pending_count"),
            "best_epoch_estimate": data.get("best_epoch_estimate"),
        } if isinstance(data, dict) else None

        images = [ProblemImage(key, entry) for key, entry in entries.items()
                  if isinstance(entry, dict)]
        images = [image for image in images if image.verdict in VERDICT_ORDER]

        # worst first, then by how far above average the image sits
        def sort_key(image: ProblemImage):
            rank = VERDICT_ORDER.index(image.verdict)
            residual = image.mean_residual if isinstance(image.mean_residual, (int, float)) else 0.0
            return rank, -residual

        images.sort(key=sort_key)
        self.images = images
        return bool(images)

    def status_text(self) -> str:
        if not os.path.exists(self.report_path):
            return ("No curation report yet. Turn on 'Detect Problem Images' on the training tab "
                    "and run at least one epoch past the warm-up.")
        if not self.images:
            return "Nothing flagged — every image is training normally."

        counts: dict[str, int] = {}
        for image in self.images:
            counts[image.verdict] = counts.get(image.verdict, 0) + 1
        summary = ", ".join(f"{counts[v]} {v}" for v in VERDICT_ORDER if v in counts)

        epoch = f" after epoch {self.generated_at}" if self.generated_at else ""
        return f"{summary}{epoch}"

    def plateau_text(self) -> str | None:
        if not isinstance(self.plateau, dict) or not self.plateau.get("plateaued"):
            return None
        best = self.plateau.get("best_epoch_estimate")
        provisional = self.plateau.get("pending")
        certainty = "provisional" if provisional else "confirmed"
        text = f"Nothing is still improving ({certainty} plateau)."
        if best:
            text += f" Learning looks finished around epoch {best} — compare checkpoints from there."
        return text

    # --- writing back ------------------------------------------------------------------------

    def save_caption(self, image: ProblemImage, caption: str) -> str | None:
        """Write an edited caption and ask the trainer to give the image a fresh start.

        Returns None on success, or a message explaining why it could not be saved. The trainer
        re-encodes it at the next epoch boundary; nothing needs restarting.
        """
        caption = (caption or "").strip()
        if not caption:
            return "The caption is empty."

        try:
            with open(image.caption_path, "w", encoding="utf-8") as f:
                f.write(caption)
        except Exception as e:
            return f"Could not write {os.path.basename(image.caption_path)}: {e}"

        return self._queue_edit(image.key)

    def _queue_edit(self, key: str) -> str | None:
        """Add a key to the pending-edits file, keeping whatever is already queued."""
        path = self.edit_queue_path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            queued = []
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, list):
                        queued = [str(k) for k in loaded]
                except Exception:
                    queued = []

            if key not in queued:
                queued.append(key)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(queued, f, indent=2)
        except Exception as e:
            return (f"The caption was saved, but the trainer could not be told to reload it ({e}). "
                    f"It will be picked up on the next run.")
        return None
