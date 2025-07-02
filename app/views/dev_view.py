"""
dev_view.py - Developer tools view for GyroSI Baby LM

System state inspection, metrics, and debugging tools.
"""

import flet as ft
from typing import Optional, Callable, Dict, Any
import json
import threading

from state import AppState
from components.common import Section, MetricCard


class DevView(ft.UserControl):
    """
    Developer view with system inspection and debugging tools.
    """

    def __init__(self, state: AppState, page: ft.Page, on_back: Callable[[], None]):
        super().__init__()
        self.state = state
        self.page = page
        self.on_back = on_back
        self.auto_refresh_enabled = True
        self.refresh_timer = None

        # Start auto-refresh
        self._start_auto_refresh()

    def build(self):
        # Header
        header = ft.Container(
            content=ft.Row(
                controls=[
                    ft.IconButton(
                        icon=ft.icons.ARROW_BACK,
                        icon_color="#0A84FF",
                        on_click=lambda _: self.on_back(),
                    ),
                    ft.Text(
                        "Developer Tools", size=20, weight=ft.FontWeight.W_600, color="#FFFFFF"
                    ),
                    ft.Container(expand=True),  # Spacer
                    ft.CupertinoSwitch(
                        value=self.auto_refresh_enabled,
                        active_color="#30D158",
                        on_change=self._toggle_auto_refresh,
                        scale=0.8,
                    ),
                    ft.Text("Auto-refresh", size=12, color="#8E8E93"),
                ]
            ),
            padding=ft.padding.all(10),
            border=ft.border.only(bottom=ft.BorderSide(1, "#38383A")),
        )

        # Get engine state
        engine_state = self.state.get_engine_state()

        # System overview metrics
        overview_section = Section(
            title="System Overview",
            controls=[
                ft.Container(
                    content=ft.Row(
                        controls=[
                            MetricCard(
                                title="Phase",
                                value=str(engine_state.get("governance", {}).get("phase", 0)),
                                subtitle="Current phase",
                                icon=ft.icons.LOOP,
                            ),
                            MetricCard(
                                title="Cycles",
                                value=str(engine_state.get("governance", {}).get("cycle_index", 0)),
                                subtitle="Completed",
                                icon=ft.icons.REFRESH,
                            ),
                        ],
                        spacing=10,
                        wrap=True,
                    ),
                    padding=ft.padding.all(10),
                )
            ],
        )

        # Governance engine details
        gov_state = engine_state.get("governance", {})
        governance_section = Section(
            title="Governance Engine",
            controls=[
                self._build_state_row("Phase", gov_state.get("phase", 0)),
                self._build_state_row("Cycle Index", gov_state.get("cycle_index", 0)),
                self._build_state_row("Buffer Size", gov_state.get("buffer_size", 0)),
                self._build_state_row("Current Cycle Size", gov_state.get("current_cycle_size", 0)),
            ],
        )

        # Information engine details
        info_state = engine_state.get("information", {})
        information_section = Section(
            title="Information Engine",
            controls=[
                self._build_state_row("Mask Shape", str(info_state.get("mask_shape", []))),
                self._build_state_row(
                    "Mask Loaded", "Yes" if info_state.get("mask_loaded") else "No"
                ),
                self._build_state_row(
                    "Resonance Ratio", f"{info_state.get('resonance_ratio', 0):.2%}"
                ),
                ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.Text("Resonance Counts", size=12, color="#8E8E93"),
                            ft.Container(
                                content=ft.Text(
                                    json.dumps(info_state.get("resonance_counts", {}), indent=2),
                                    size=11,
                                    color="#FFFFFF",
                                    font_family="monospace",
                                ),
                                bgcolor="#000000",
                                padding=ft.padding.all(10),
                                border_radius=8,
                            ),
                        ],
                        spacing=5,
                    ),
                    padding=ft.padding.symmetric(horizontal=20, vertical=10),
                ),
            ],
        )

        # Inference engine details
        inf_state = engine_state.get("inference", {})
        inference_section = Section(
            title="Inference Engine",
            controls=[
                self._build_state_row("Promoted Patterns", inf_state.get("promoted_patterns", 0)),
                self._build_state_row("Cycle History Size", inf_state.get("cycle_history_size", 0)),
                ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.Text("Prune Statistics", size=12, color="#8E8E93"),
                            ft.Container(
                                content=ft.Text(
                                    json.dumps(inf_state.get("prune_stats", {}), indent=2),
                                    size=11,
                                    color="#FFFFFF",
                                    font_family="monospace",
                                ),
                                bgcolor="#000000",
                                padding=ft.padding.all(10),
                                border_radius=8,
                            ),
                        ],
                        spacing=5,
                    ),
                    padding=ft.padding.symmetric(horizontal=20, vertical=10),
                ),
                ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.Text("Top Patterns", size=12, color="#8E8E93"),
                            ft.Container(
                                content=ft.Text(
                                    json.dumps(inf_state.get("top_patterns", {}), indent=2),
                                    size=11,
                                    color="#FFFFFF",
                                    font_family="monospace",
                                ),
                                bgcolor="#000000",
                                padding=ft.padding.all(10),
                                border_radius=8,
                            ),
                        ],
                        spacing=5,
                    ),
                    padding=ft.padding.symmetric(horizontal=20, vertical=10),
                ),
            ],
        )

        # Processing stats from last operation
        processing_section = Section(
            title="Last Processing Stats",
            controls=[
                ft.Container(
                    content=ft.Text(
                        (
                            json.dumps(self.state.processing_stats, indent=2)
                            if self.state.processing_stats
                            else "No recent processing"
                        ),
                        size=11,
                        color="#FFFFFF",
                        font_family="monospace",
                    ),
                    bgcolor="#000000",
                    padding=ft.padding.all(10),
                    border_radius=8,
                    width=float("inf"),
                )
            ],
        )

        # Scroll view with all sections
        return ft.Column(
            controls=[
                header,
                ft.Container(
                    content=ft.Column(
                        controls=[
                            overview_section,
                            governance_section,
                            information_section,
                            inference_section,
                            processing_section,
                        ],
                        spacing=20,
                        scroll=ft.ScrollMode.AUTO,
                    ),
                    expand=True,
                    padding=ft.padding.all(20),
                ),
            ],
            spacing=0,
            expand=True,
        )

    def _build_state_row(self, label: str, value: Any) -> ft.Container:
        """Build a state display row."""
        return ft.Container(
            content=ft.Row(
                controls=[
                    ft.Text(label, size=13, color="#8E8E93"),
                    ft.Container(expand=True),  # Spacer
                    ft.Text(str(value), size=13, color="#FFFFFF", weight=ft.FontWeight.W_500),
                ]
            ),
            padding=ft.padding.symmetric(horizontal=20, vertical=8),
            border=ft.border.only(bottom=ft.BorderSide(1, "#1C1C1E")),
        )

    def _toggle_auto_refresh(self, e):
        """Toggle auto-refresh."""
        self.auto_refresh_enabled = e.control.value
        if self.auto_refresh_enabled:
            self._start_auto_refresh()
        else:
            self._stop_auto_refresh()

    def _start_auto_refresh(self):
        """Start auto-refresh timer."""
        self._stop_auto_refresh()  # Ensure no duplicate timers

        def refresh_callback():
            if self.auto_refresh_enabled:
                self.update()
                self.refresh_timer = threading.Timer(1.0, refresh_callback)
                self.refresh_timer.daemon = True
                self.refresh_timer.start()

        self.refresh_timer = threading.Timer(1.0, refresh_callback)
        self.refresh_timer.daemon = True
        self.refresh_timer.start()

    def _stop_auto_refresh(self):
        """Stop auto-refresh timer."""
        if self.refresh_timer:
            self.refresh_timer.cancel()
            self.refresh_timer = None

    def will_unmount(self):
        """Clean up when view is unmounted."""
        self._stop_auto_refresh()
