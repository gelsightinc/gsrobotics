import os

os.environ["KIVY_NO_ARGS"] = "1"
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.config import Config
from kivy.metrics import dp
from kivy.graphics.texture import Texture

from utilities.gelsightmini import GelSightMini
from utilities.image_processing import add_fps_count_overlay, rescale
from utilities.ui_components import ConnectingOverlay, FileChooserPopup, DualTopBar
from config import ConfigModel
from utilities.logger import log_message

Config.set("input", "mouse", "mouse,multitouch_on_demand")


class DualGelsightMini(App):
    def __init__(self, config: ConfigModel, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.cam_stream1 = GelSightMini(
            target_width=self.config.camera_width,
            target_height=self.config.camera_height,
            border_fraction=self.config.border_fraction,
        )
        self.cam_stream2 = GelSightMini(
            target_width=self.config.second_camera_width,
            target_height=self.config.second_camera_height,
            border_fraction=self.config.border_fraction,
        )

    def build(self):
        self.title = "Gelsight Mini Dual Viewer"
        self.loading_overlay = None
        root = BoxLayout(orientation="vertical")
        self.top_bar = DualTopBar(
            on_device_selected_callback=self.on_dual_device_selected
        )
        root.add_widget(self.top_bar)
        self.live_view = DualLiveViewWidget(main_app=self)
        root.add_widget(self.live_view)
        return root

    def show_overlay(self, message):
        if not self.loading_overlay:
            self.loading_overlay = ConnectingOverlay(message=message)
            self.loading_overlay.open()

    def hide_overlay(self):
        if self.loading_overlay:
            self.loading_overlay.dismiss()
            self.loading_overlay = None

    def on_dual_device_selected(self, device_indices):
        # Only start feeds if both device indices are valid and not the default text
        idx1, idx2 = (
            device_indices if isinstance(device_indices, (list, tuple)) else (0, 1)
        )
        if isinstance(idx1, int) and isinstance(idx2, int) and idx1 != idx2:
            self.restart_camera_streams((idx1, idx2))

    def restart_camera_streams(self, device_indices):
        # device_indices should be a tuple/list: (idx1, idx2)
        idx1, idx2 = (
            device_indices if isinstance(device_indices, (list, tuple)) else (0, 1)
        )
        self.cam_stream1.select_device(idx1)
        self.cam_stream2.select_device(idx2)
        Clock.schedule_once(lambda dt: self.finish_device_selection(), 0)

    def finish_device_selection(self):
        self.hide_overlay()
        self.cam_stream1.start()
        self.cam_stream2.start()
        self.live_view.start()


class DualLiveViewWidget(BoxLayout):
    def __init__(self, main_app: DualGelsightMini, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.main_app = main_app
        images_layout = BoxLayout(orientation="horizontal", size_hint_y=0.7)
        self.image_widget1 = Image()
        self.image_widget2 = Image()
        images_layout.add_widget(self.image_widget1)
        images_layout.add_widget(self.image_widget2)
        self.add_widget(images_layout)
        zoom_layout = BoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(40)
        )
        zoom_layout.add_widget(
            Label(text="Zoom:", size_hint=(None, None), size=(dp(60), dp(40)))
        )
        self.zoom_slider = Slider(min=0.5, max=3.0, value=1.0)
        self.zoom_slider.bind(value=self.on_zoom_value_change)
        zoom_layout.add_widget(self.zoom_slider)
        self.zoom_label = Label(
            text="1.0x", size_hint=(None, None), size=(dp(60), dp(40))
        )
        zoom_layout.add_widget(self.zoom_label)
        self.add_widget(zoom_layout)
        folder_layout = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(40),
            spacing=dp(80),
            padding=[dp(10)] * 4,
        )
        self.screenshot_folder_btn = Button(
            text="Data Folder", size_hint=(None, None), size=(dp(180), dp(30))
        )
        self.screenshot_folder_btn.bind(on_press=self.open_screenshot_folder_choice)
        folder_layout.add_widget(self.screenshot_folder_btn)
        self.screenshot_folder_label = Label(
            text=f"{os.path.join(os.path.expanduser('~'), 'Desktop')}",
            size_hint=(None, None),
            height=dp(30),
            width=dp(400),
            halign="left",
            valign="middle",
        )
        folder_layout.add_widget(self.screenshot_folder_label)
        self.add_widget(folder_layout)
        capture_layout = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(40),
            spacing=dp(20),
            padding=[dp(10)] * 4,
        )
        self.screenshot_btn = Button(
            text="Save Images", size_hint=(None, None), size=(dp(180), dp(30))
        )
        self.screenshot_btn.bind(on_press=self.take_screenshots)
        capture_layout.add_widget(self.screenshot_btn)
        self.recording_btn = Button(
            text="Start Recording", size_hint=(None, None), size=(dp(180), dp(30))
        )
        self.recording_btn.bind(on_press=self.recording)
        capture_layout.add_widget(self.recording_btn)
        self.add_widget(capture_layout)
        self.add_widget(Widget(size_hint_y=None, height=dp(10)))
        self.screenshot_folder = os.path.join(os.path.expanduser("~"), "Desktop")
        self.event = None
        Window.bind(on_key_down=self.on_key_down)

    def on_zoom_value_change(self, instance, value):
        self.zoom_label.text = f"{value:.1f}x"

    def start(self):
        self.main_app.cam_stream1.start()
        self.main_app.cam_stream2.start()
        self.event = Clock.schedule_interval(self.update, 1 / 30.0)

    def update(self, dt):
        for idx, (cam_stream, image_widget) in enumerate(
            [
                (self.main_app.cam_stream1, self.image_widget1),
                (self.main_app.cam_stream2, self.image_widget2),
            ],
            start=1,
        ):
            frame = cam_stream.update(dt)
            if frame is not None:
                scale = self.zoom_slider.value
                if scale != 1.0:
                    frame = rescale(frame, scale=scale)
                add_fps_count_overlay(frame=frame, fps=cam_stream.fps)
                texture = Texture.create(
                    size=(frame.shape[1], frame.shape[0]), colorfmt="rgb"
                )
                texture.blit_buffer(frame.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
                texture.flip_vertical()
                image_widget.texture = texture

    def stop(self):
        if self.event:
            self.event.cancel()
        for cam_stream in [self.main_app.cam_stream1, self.main_app.cam_stream2]:
            if cam_stream.camera:
                cam_stream.camera.release()

    def take_screenshots(self, instance=None):
        for idx, cam_stream in enumerate(
            [self.main_app.cam_stream1, self.main_app.cam_stream2], start=1
        ):
            folder = os.path.join(self.screenshot_folder, f"cam{idx}")
            os.makedirs(folder, exist_ok=True)
            cam_stream.save_screenshot(filepath=folder)

    def recording(self, instance=None):
        if self.main_app.cam_stream1.recording or self.main_app.cam_stream2.recording:
            self.main_app.cam_stream1.stop_recording()
            self.main_app.cam_stream2.stop_recording()
            self.recording_btn.text = "Start Recording"
        else:
            for idx, cam_stream in enumerate(
                [self.main_app.cam_stream1, self.main_app.cam_stream2], start=1
            ):
                folder = os.path.join(self.screenshot_folder, f"cam{idx}")
                os.makedirs(folder, exist_ok=True)
                cam_stream.start_recording(filepath=folder)
            self.recording_btn.text = "Stop Recording"

    def on_key_down(self, window, key, *args):
        if key == 32:
            self.take_screenshots()

    def open_screenshot_folder_choice(self, instance):
        popup = FileChooserPopup(self.select_screenshot_folder)
        popup.open()

    def select_screenshot_folder(self, path):
        if path:
            self.screenshot_folder = path
            self.screenshot_folder_label.text = (
                f"Target Folder: {self.screenshot_folder}"
            )


if __name__ == "__main__":
    import argparse
    from config import GSConfig

    parser = argparse.ArgumentParser(
        description="Run the Gelsight Mini Dual Viewer with an optional config file."
    )
    parser.add_argument(
        "--gs-config",
        type=str,
        default=None,
        help="Path to the JSON configuration file. If not provided, default config is used.",
    )
    args = parser.parse_args()
    if args.gs_config is not None:
        log_message(f"Provided config path: {args.gs_config}")
    else:
        log_message(f"Didn't provide custom config path.")
        log_message(
            f"Using default config path './default_config.json' if such file exists."
        )
        log_message(
            f"Using default_config variable from 'config.py' if './default_config.json' is not available"
        )
        args.gs_config = "default_config.json"
    gs_config = GSConfig(args.gs_config)
    DualGelsightMini(config=gs_config.config).run()
