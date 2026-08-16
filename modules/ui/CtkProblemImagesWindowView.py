from modules.ui.ProblemImagesWindowController import (
    VERDICT_HELP,
    ProblemImage,
    ProblemImagesWindowController,
)
from modules.util.image_util import load_image
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk
from PIL import Image

THUMBNAIL_SIZE = 128
REFRESH_MS = 4000

VERDICT_COLOURS = {
    "excluded": "#6c757d",
    "stuck": "#dc3545",
    "suspect": "#fd7e14",
    "exhausted": "#0d6efd",
    "watch": "#ffc107",
    "learning": "#198754",
}


class CtkProblemImagesWindowView(ctk.CTkToplevel):
    def __init__(self, parent, controller: ProblemImagesWindowController):
        super().__init__(parent)

        self.controller = controller
        self._thumbnails: list[ctk.CTkImage] = []
        self._editors: dict[str, ctk.CTkTextbox] = {}
        self._refresh_job = None

        self.title("Problem Images")
        self.geometry("980x760")
        self.resizable(True, True)
        self.after(200, lambda: set_window_icon(self))

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, corner_radius=0)
        header.grid(row=0, column=0, sticky="new")
        header.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(header, text="", anchor="w", justify="left")
        self.status_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")

        self.plateau_label = ctk.CTkLabel(header, text="", anchor="w", justify="left",
                                          text_color="#0d6efd")
        self.plateau_label.grid(row=1, column=0, padx=10, pady=(2, 10), sticky="ew")

        ctk.CTkButton(header, text="Refresh", width=90, command=self.refresh) \
            .grid(row=0, column=1, rowspan=2, padx=10, pady=10)

        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.scroll.grid_columnconfigure(1, weight=1)

        self.protocol("WM_DELETE_WINDOW", self.__close)
        self.refresh()
        self.__schedule_refresh()

    # --- lifecycle ---------------------------------------------------------------------------

    def __schedule_refresh(self):
        # the trainer rewrites the report once per epoch; polling is how the window notices
        self._refresh_job = self.after(REFRESH_MS, self.__auto_refresh)

    def __auto_refresh(self):
        try:
            if self.controller.has_changed():
                self.refresh()
        finally:
            self.__schedule_refresh()

    def __close(self):
        if self._refresh_job is not None:
            self.after_cancel(self._refresh_job)
            self._refresh_job = None
        self.destroy()

    # --- rendering ---------------------------------------------------------------------------

    def refresh(self):
        # Unsaved edits belong to the user, not the report; keep them across a refresh so a
        # boundary landing mid-sentence does not wipe what they were typing.
        pending = {key: box.get("1.0", "end").strip() for key, box in self._editors.items()}

        self.controller.load()

        for child in self.scroll.winfo_children():
            child.destroy()
        self._thumbnails.clear()
        self._editors.clear()

        self.status_label.configure(text=self.controller.status_text())
        plateau = self.controller.plateau_text()
        self.plateau_label.configure(text=plateau or "")

        for row, image in enumerate(self.controller.images):
            self.__build_row(row, image, pending.get(image.key))

    def __build_row(self, row: int, image: ProblemImage, pending_caption: str | None):
        frame = ctk.CTkFrame(self.scroll)
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        frame.grid_columnconfigure(1, weight=1)

        thumbnail = ctk.CTkLabel(frame, text="", width=THUMBNAIL_SIZE, height=THUMBNAIL_SIZE)
        thumbnail.grid(row=0, column=0, rowspan=3, padx=8, pady=8)
        picture = self.__load_thumbnail(image)
        if picture is not None:
            self._thumbnails.append(picture)
            thumbnail.configure(image=picture)
        else:
            thumbnail.configure(text="(missing)")

        ctk.CTkLabel(frame, text=image.name, anchor="w", font=ctk.CTkFont(weight="bold")) \
            .grid(row=0, column=1, sticky="ew", padx=4, pady=(8, 0))

        ctk.CTkLabel(
            frame, text=image.summary(), anchor="w",
            text_color=VERDICT_COLOURS.get(image.verdict),
        ).grid(row=1, column=1, sticky="ew", padx=4)

        ctk.CTkLabel(frame, text=VERDICT_HELP.get(image.verdict, ""), anchor="w",
                     wraplength=620, justify="left").grid(row=2, column=1, sticky="ew", padx=4)

        editor = ctk.CTkTextbox(frame, height=64, wrap="word")
        editor.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 4))
        editor.insert("1.0", pending_caption if pending_caption is not None else image.read_caption())
        self._editors[image.key] = editor

        actions = ctk.CTkFrame(frame, fg_color="transparent")
        actions.grid(row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

        message = ctk.CTkLabel(actions, text="", anchor="w")
        message.grid(row=0, column=1, sticky="ew", padx=8)
        actions.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            actions, text="Save caption", width=120,
            command=lambda i=image, e=editor, m=message: self.__save(i, e, m),
        ).grid(row=0, column=0)

    def __save(self, image: ProblemImage, editor: ctk.CTkTextbox, message: ctk.CTkLabel):
        error = self.controller.save_caption(image, editor.get("1.0", "end"))
        if error:
            message.configure(text=error, text_color="#dc3545")
        else:
            message.configure(text="Saved — the trainer picks it up at the next epoch boundary.",
                              text_color="#198754")

    def __load_thumbnail(self, image: ProblemImage):
        if not image.exists:
            return None
        try:
            picture = load_image(image.key, convert_mode="RGB")
        except Exception:
            return None
        picture.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.BILINEAR)
        return ctk.CTkImage(light_image=picture, dark_image=picture, size=picture.size)
