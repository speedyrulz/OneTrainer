from modules.ui.ProblemImagesWindowController import (
    VERDICT_HELP,
    ProblemImage,
    ProblemImagesWindowController,
)

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

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


class PySide6ProblemImagesWindowView(QWidget):
    def __init__(self, parent, controller: ProblemImagesWindowController):
        super().__init__(parent, Qt.WindowType.Window)

        self.controller = controller
        self._editors: dict[str, QPlainTextEdit] = {}

        self.setWindowTitle("Problem Images")
        self.setWindowIcon(QIcon("resources/icons/icon.png"))
        self.resize(980, 760)

        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        labels = QVBoxLayout()

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        labels.addWidget(self.status_label)

        self.plateau_label = QLabel("")
        self.plateau_label.setWordWrap(True)
        self.plateau_label.setStyleSheet("color: #0d6efd;")
        labels.addWidget(self.plateau_label)

        header.addLayout(labels, 1)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        header.addWidget(refresh_button, 0, Qt.AlignmentFlag.AlignTop)

        layout.addLayout(header)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll, 1)

        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.addStretch(1)
        self.scroll.setWidget(self.container)

        self.refresh()

        # the trainer rewrites the report once per epoch; polling is how the window notices
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.__auto_refresh)
        self._timer.start(REFRESH_MS)

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)

    # --- rendering ---------------------------------------------------------------------------

    def __auto_refresh(self):
        if self.controller.has_changed():
            self.refresh()

    def refresh(self):
        # Unsaved edits belong to the user, not the report; keep them across a refresh so a
        # boundary landing mid-sentence does not wipe what they were typing.
        pending = {key: box.toPlainText().strip() for key, box in self._editors.items()}

        self.controller.load()
        self._editors.clear()

        while self.container_layout.count() > 1:
            item = self.container_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        self.status_label.setText(self.controller.status_text())
        self.plateau_label.setText(self.controller.plateau_text() or "")

        for index, image in enumerate(self.controller.images):
            row = self.__build_row(image, pending.get(image.key))
            self.container_layout.insertWidget(index, row)

    def __build_row(self, image: ProblemImage, pending_caption: str | None) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(frame)
        grid.setColumnStretch(1, 1)

        thumbnail = QLabel()
        thumbnail.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = self.__load_thumbnail(image)
        if pixmap is not None:
            thumbnail.setPixmap(pixmap)
        else:
            thumbnail.setText("(missing)")
        grid.addWidget(thumbnail, 0, 0, 3, 1)

        name = QLabel(image.name)
        name.setStyleSheet("font-weight: bold;")
        grid.addWidget(name, 0, 1)

        summary = QLabel(image.summary())
        colour = VERDICT_COLOURS.get(image.verdict)
        if colour:
            summary.setStyleSheet(f"color: {colour};")
        grid.addWidget(summary, 1, 1)

        help_text = QLabel(VERDICT_HELP.get(image.verdict, ""))
        help_text.setWordWrap(True)
        grid.addWidget(help_text, 2, 1)

        editor = QPlainTextEdit()
        editor.setFixedHeight(72)
        editor.setPlainText(
            pending_caption if pending_caption is not None else image.read_caption()
        )
        grid.addWidget(editor, 3, 0, 1, 2)
        self._editors[image.key] = editor

        actions = QHBoxLayout()
        save_button = QPushButton("Save caption")
        actions.addWidget(save_button)

        message = QLabel("")
        message.setWordWrap(True)
        actions.addWidget(message, 1)
        grid.addLayout(actions, 4, 0, 1, 2)

        save_button.clicked.connect(
            lambda _checked=False, i=image, e=editor, m=message: self.__save(i, e, m)
        )

        return frame

    def __save(self, image: ProblemImage, editor: QPlainTextEdit, message: QLabel):
        error = self.controller.save_caption(image, editor.toPlainText())
        if error:
            message.setText(error)
            message.setStyleSheet("color: #dc3545;")
        else:
            message.setText("Saved — the trainer picks it up at the next epoch boundary.")
            message.setStyleSheet("color: #198754;")

    @staticmethod
    def __load_thumbnail(image: ProblemImage) -> QPixmap | None:
        if not image.exists:
            return None
        pixmap = QPixmap(image.key)
        if pixmap.isNull():
            return None
        return pixmap.scaled(
            THUMBNAIL_SIZE, THUMBNAIL_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
