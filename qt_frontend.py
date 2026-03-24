from __future__ import annotations

import sys

import numpy as np
from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from game.core import Game
from game.events import GameWonEvent, PlayerGameOverEvent
from game.interest import hard_cap, interest_rate_per_tick, soft_cap


class ClickableMapLabel(QLabel):
    clicked = pyqtSignal(QPoint, QPoint)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event.position().toPoint(), event.globalPosition().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)


class GameWindow(QMainWindow):
    ATTACK_PERCENTAGE = 0.25
    MIN_TICK_INTERVAL_MS = 50
    MAX_TICK_INTERVAL_MS = 2000

    def __init__(self) -> None:
        super().__init__()

        self.game = Game(seed=42)
        self.player2_id = self.game.add_player()
        if self.player2_id != 2:
            raise RuntimeError(f"Expected first spawned player to be 2, got {self.player2_id}.")

        self.setWindowTitle("Territory Bot Viewer")
        self.resize(1100, 760)

        self._selected_target: tuple[int, int, int] | None = None

        self._map_label = ClickableMapLabel()
        self._map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._map_label.setMinimumSize(512, 512)
        self._map_label.setStyleSheet(
            "background: #101418; border: 1px solid #2b3640; padding: 8px;"
        )
        self._map_label.clicked.connect(self._on_map_clicked)

        self._tick_value = QLabel()
        self._tick_speed_value = QLabel()
        self._winner_value = QLabel()
        self._occupiable_area_value = QLabel()
        self._alive_players_value = QLabel()

        self._player_alive_value = QLabel()
        self._player_balance_value = QLabel()
        self._player_balance_cap_value = QLabel()
        self._player_interest_value = QLabel()
        self._player_area_value = QLabel()
        self._player_spawn_value = QLabel()
        self._player_occupation_value = QLabel()
        self._player_eliminated_value = QLabel()
        self._player_game_over_value = QLabel()
        self._player_game_won_value = QLabel()

        self._event_list = QListWidget()
        self._event_list.setAlternatingRowColors(True)

        self._tick_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._tick_speed_slider.setRange(
            self.MIN_TICK_INTERVAL_MS, self.MAX_TICK_INTERVAL_MS
        )
        self._tick_speed_slider.setSingleStep(50)
        self._tick_speed_slider.setPageStep(100)
        self._tick_speed_slider.setValue(500)
        self._tick_speed_slider.valueChanged.connect(self._on_tick_speed_changed)

        self._attack_popup = QFrame(self)
        self._attack_popup.setFrameShape(QFrame.Shape.StyledPanel)
        self._attack_popup.setStyleSheet(
            "QFrame { background: #1a222a; border: 1px solid #3f4d5a; }"
            "QLabel { color: #d9e2ea; }"
            "QPushButton { padding: 6px 10px; }"
        )
        self._attack_popup.hide()
        popup_layout = QVBoxLayout(self._attack_popup)
        popup_layout.setContentsMargins(10, 10, 10, 10)
        popup_layout.setSpacing(8)
        self._attack_popup_title = QLabel()
        self._attack_popup_button = QPushButton(
            f"Attack ({int(self.ATTACK_PERCENTAGE * 100)}%)"
        )
        self._attack_popup_button.clicked.connect(self._queue_attack_from_popup)
        popup_layout.addWidget(self._attack_popup_title)
        popup_layout.addWidget(self._attack_popup_button)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._refresh_view()
        self._timer.start()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)

        left_column = QVBoxLayout()
        left_column.addWidget(self._map_label, stretch=1)

        controls = QHBoxLayout()
        pause_button = QPushButton("Pause")
        pause_button.clicked.connect(self._pause)
        resume_button = QPushButton("Resume")
        resume_button.clicked.connect(self._resume)
        spawn_button = QPushButton("Spawn Player")
        spawn_button.clicked.connect(self._spawn_player)
        controls.addWidget(pause_button)
        controls.addWidget(resume_button)
        controls.addWidget(spawn_button)
        left_column.addLayout(controls)

        speed_controls = QHBoxLayout()
        speed_label = QLabel("Tick Speed")
        speed_label.setStyleSheet("font-weight: 600;")
        speed_controls.addWidget(speed_label)
        speed_controls.addWidget(self._tick_speed_slider, stretch=1)
        speed_controls.addWidget(self._tick_speed_value)
        left_column.addLayout(speed_controls)

        right_column = QVBoxLayout()
        right_column.addWidget(self._build_game_box())
        right_column.addWidget(self._build_player_box())
        right_column.addWidget(self._build_event_box(), stretch=1)

        layout.addLayout(left_column, stretch=3)
        layout.addLayout(right_column, stretch=2)
        self.setCentralWidget(root)

    def _build_game_box(self) -> QGroupBox:
        box = QGroupBox("Game")
        grid = QGridLayout(box)
        self._add_stat_row(grid, 0, "Tick", self._tick_value)
        self._add_stat_row(grid, 1, "Winner", self._winner_value)
        self._add_stat_row(grid, 2, "Occupiable Area", self._occupiable_area_value)
        self._add_stat_row(grid, 3, "Alive Players", self._alive_players_value)
        return box

    def _build_player_box(self) -> QGroupBox:
        box = QGroupBox("Player 2")
        grid = QGridLayout(box)
        self._add_stat_row(grid, 0, "Alive", self._player_alive_value)
        self._add_stat_row(grid, 1, "Balance", self._player_balance_value)
        self._add_stat_row(grid, 2, "Balance Cap", self._player_balance_cap_value)
        self._add_stat_row(grid, 3, "Interest", self._player_interest_value)
        self._add_stat_row(grid, 4, "Owned Tiles", self._player_area_value)
        self._add_stat_row(grid, 5, "Occupation", self._player_occupation_value)
        self._add_stat_row(grid, 6, "Spawn", self._player_spawn_value)
        self._add_stat_row(grid, 7, "Eliminated Tick", self._player_eliminated_value)
        self._add_stat_row(grid, 8, "Game Over", self._player_game_over_value)
        self._add_stat_row(grid, 9, "Game Won", self._player_game_won_value)
        return box

    def _build_event_box(self) -> QGroupBox:
        box = QGroupBox("Recent Events")
        layout = QVBoxLayout(box)
        layout.addWidget(self._event_list)
        return box

    def _add_stat_row(
        self, layout: QGridLayout, row: int, label_text: str, value_label: QLabel
    ) -> None:
        name = QLabel(label_text)
        name.setStyleSheet("font-weight: 600;")
        value_label.setText("-")
        layout.addWidget(name, row, 0)
        layout.addWidget(value_label, row, 1)

    def _pause(self) -> None:
        self._timer.stop()

    def _resume(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def _on_tick_speed_changed(self, value: int) -> None:
        self._timer.setInterval(value)
        self._tick_speed_value.setText(f"{value} ms")
        self._refresh_view()

    def _spawn_player(self) -> None:
        player_id = self.game.add_player()
        if player_id == -1:
            self._prepend_event_text("No unoccupied land left to spawn a new player.")
        else:
            self._prepend_event_text(f"Spawned player {player_id}.")
        self._refresh_view()

    def _on_tick(self) -> None:
        events = self.game.tick()
        for event in events:
            self._prepend_event_text(self._format_event(event))
        self._refresh_view()

    def _on_map_clicked(self, local_pos: QPoint, global_pos: QPoint) -> None:
        clicked_cell = self._map_position_to_cell(local_pos)
        if clicked_cell is None:
            self._hide_attack_popup()
            return

        row, col = clicked_cell
        owner_id = int(self.game.map[row, col])
        if owner_id in (0, self.player2_id):
            self._hide_attack_popup()
            return

        self._selected_target = (row, col, owner_id)
        owner_text = "neutral land" if owner_id == 1 else f"player {owner_id}"
        self._attack_popup_title.setText(f"({row}, {col})\nTarget: {owner_text}")
        self._attack_popup.adjustSize()
        self._attack_popup.move(self.mapFromGlobal(global_pos))
        self._attack_popup.show()
        self._attack_popup.raise_()

    def _map_position_to_cell(self, local_pos: QPoint) -> tuple[int, int] | None:
        pixmap = self._map_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return None

        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        if pixmap_width <= 0 or pixmap_height <= 0:
            return None

        x_offset = (self._map_label.width() - pixmap_width) // 2
        y_offset = (self._map_label.height() - pixmap_height) // 2

        if not (
            x_offset <= local_pos.x() < x_offset + pixmap_width
            and y_offset <= local_pos.y() < y_offset + pixmap_height
        ):
            return None

        map_height, map_width = self.game.map.shape
        col = int((local_pos.x() - x_offset) * map_width / pixmap_width)
        row = int((local_pos.y() - y_offset) * map_height / pixmap_height)
        row = max(0, min(map_height - 1, row))
        col = max(0, min(map_width - 1, col))
        return row, col

    def _queue_attack_from_popup(self) -> None:
        if self._selected_target is None:
            self._hide_attack_popup()
            return

        row, col, owner_id = self._selected_target
        queued = self.game.attack(
            self.player2_id,
            {
                "type": "attack",
                "target": [row, col],
                "percentage": self.ATTACK_PERCENTAGE,
            },
        )
        if queued:
            owner_text = "neutral land" if owner_id == 1 else f"player {owner_id}"
            self._prepend_event_text(
                f"Queued attack from player {self.player2_id} onto {owner_text} at "
                f"({row}, {col}) with {int(self.ATTACK_PERCENTAGE * 100)}% balance."
            )
        else:
            self._prepend_event_text(
                f"Attack from player {self.player2_id} onto ({row}, {col}) could not be queued."
            )

        self._hide_attack_popup()
        self._refresh_view()

    def _hide_attack_popup(self) -> None:
        self._selected_target = None
        self._attack_popup.hide()

    def _refresh_view(self) -> None:
        state = self.game.get_state(relative=self.player2_id)
        ownership_map = state[0]

        self._tick_value.setText(str(self.game.tick_count))
        self._winner_value.setText(
            "None" if self.game.winner_id is None else str(self.game.winner_id)
        )
        self._occupiable_area_value.setText(str(self.game._occupiable_area))
        alive_players = sum(1 for player in self.game.players.values() if player.is_alive)
        self._alive_players_value.setText(str(alive_players))

        player = self.game.players[self.player2_id]
        owned_tiles = int(np.count_nonzero(self.game.map == self.player2_id))
        occupation = (
            0.0
            if self.game._occupiable_area <= 0
            else owned_tiles / self.game._occupiable_area
        )
        interest_rate = interest_rate_per_tick(
            balance=player.balance,
            owned_area=owned_tiles,
            occupiable_area=self.game._occupiable_area,
            tick=self.game.tick_count,
        )
        soft_balance_cap = soft_cap(owned_tiles)
        hard_balance_cap = hard_cap(owned_tiles)
        self._player_alive_value.setText("yes" if player.is_alive else "no")
        self._player_balance_value.setText(str(player.balance))
        self._player_balance_cap_value.setText(
            f"{soft_balance_cap} soft / {hard_balance_cap} hard"
        )
        self._player_interest_value.setText(f"{interest_rate:.2%}/tick")
        self._player_area_value.setText(str(owned_tiles))
        self._player_occupation_value.setText(f"{occupation:.2%}")
        self._player_spawn_value.setText(f"({player.spawn_row}, {player.spawn_col})")
        self._player_eliminated_value.setText(
            "-" if player.eliminated_tick is None else str(player.eliminated_tick)
        )
        self._player_game_over_value.setText("yes" if not player.is_alive else "no")
        self._player_game_won_value.setText(
            "yes" if self.game.winner_id == self.player2_id else "no"
        )

        self._map_label.setPixmap(
            QPixmap.fromImage(_ownership_map_to_qimage(ownership_map)).scaled(
                self._map_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )

    def _prepend_event_text(self, text: str) -> None:
        self._event_list.insertItem(0, QListWidgetItem(text))
        while self._event_list.count() > 24:
            self._event_list.takeItem(self._event_list.count() - 1)

    def _format_event(self, event: PlayerGameOverEvent | GameWonEvent) -> str:
        if isinstance(event, PlayerGameOverEvent):
            return f"Tick {event.tick}: player {event.player_id} was eliminated."
        if isinstance(event, GameWonEvent):
            return (
                f"Tick {event.tick}: player {event.player_id} won at "
                f"{event.occupation_fraction:.2%} occupation."
            )
        return repr(event)

    def resizeEvent(self, event) -> None:
        self._hide_attack_popup()
        super().resizeEvent(event)

    def changeEvent(self, event) -> None:
        if event.type() == event.Type.ActivationChange and not self.isActiveWindow():
            self._hide_attack_popup()
        super().changeEvent(event)


def _ownership_map_to_qimage(ownership_map: np.ndarray) -> QImage:
    rgb = np.zeros((*ownership_map.shape, 3), dtype=np.uint8)

    rgb[ownership_map == 0] = (25, 64, 99)
    rgb[ownership_map == 1] = (184, 168, 121)

    occupied = ownership_map >= 2
    occupied_ids = ownership_map[occupied].astype(np.uint32, copy=False)
    if occupied_ids.size:
        rgb[occupied, 0] = ((occupied_ids * 97) % 215 + 40).astype(np.uint8)
        rgb[occupied, 1] = ((occupied_ids * 57) % 215 + 40).astype(np.uint8)
        rgb[occupied, 2] = ((occupied_ids * 23) % 215 + 40).astype(np.uint8)
        rgb[ownership_map == 2] = (230, 88, 62)

    image = QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    )
    return image.copy()


def main() -> int:
    app = QApplication(sys.argv)
    window = GameWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
