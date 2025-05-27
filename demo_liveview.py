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
from utilities.ui_components import ConnectingOverlay, FileChooserPopup, TopBar
from config import ConfigModel
from utilities.logger import log_message

Config.set("input", "mouse", "mouse,multitouch_on_demand")


class GelsightMini(App):
    def __init__(self, config: ConfigModel, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.cam_stream = GelSightMini(
            target_width=self.config.camera_width,
            target_height=self.config.camera_height,
            border_fraction=self.config.border_fraction,
        )

    def build(self):
        self.title = "Gelsight Mini Viewer"
        self.loading_overlay = None

        root = BoxLayout(orientation="vertical")
        # Top bar with device selection
        self.top_bar = TopBar(on_device_selected_callback=self.restart_camera_stream)
        root.add_widget(self.top_bar)

        # Create LiveViewWidget and pass the app instance so it has access to camstream and other properties.
        self.live_view = LiveViewWidget(main_app=self)
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

    def restart_camera_stream(self, device_index):
        self.cam_stream.select_device(device_index)
        from kivy.clock import Clock

        Clock.schedule_once(lambda dt: self.finish_device_selection(), 0)

    def finish_device_selection(self):
        self.hide_overlay()
        self.cam_stream.start()
        self.live_view.start()


class LiveViewWidget(BoxLayout):
    def __init__(self, main_app: GelsightMini, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.main_app = main_app

        # Create UI elements:
        self.image_widget = Image()
        self.add_widget(self.image_widget)

        # Zoom slider layout
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

        # Folder selection layout
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

        # Capture controls layout
        capture_layout = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(40),
            spacing=dp(20),
            padding=[dp(10)] * 4,
        )
        self.screenshot_btn = Button(
            text="Save Image", size_hint=(None, None), size=(dp(180), dp(30))
        )
        self.screenshot_btn.bind(on_press=self.take_screenshot)
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

        # Bind key events for shortcuts
        Window.bind(on_key_down=self.on_key_down)

    def on_zoom_value_change(self, instance, value):
        self.zoom_label.text = f"{value:.1f}x"

    def start(self):
        # Start the camera stream and schedule updates.
        self.main_app.cam_stream.start()
        self.event = Clock.schedule_interval(self.update, 1 / 30.0)

    def update(self, dt):
        frame = self.main_app.cam_stream.update(dt)
        if frame is not None:
            scale = self.zoom_slider.value
            if scale != 1.0:
                frame = rescale(frame, scale=scale)
            add_fps_count_overlay(frame=frame, fps=self.main_app.cam_stream.fps)

            texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt="rgb"
            )
            texture.blit_buffer(frame.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
            texture.flip_vertical()
            self.image_widget.texture = texture

    def stop(self):
        if self.event:
            self.event.cancel()
        if self.main_app.cam_stream.camera:
            self.main_app.cam_stream.camera.release()

    def take_screenshot(self, instance=None):
        self.main_app.cam_stream.save_screenshot(filepath=self.screenshot_folder)

    def recording(self, instance=None):
        if self.main_app.cam_stream.recording:
            self.main_app.cam_stream.stop_recording()
            self.recording_btn.text = "Start Recording"
        else:
            self.main_app.cam_stream.start_recording(filepath=self.screenshot_folder)
            self.recording_btn.text = "Stop Recording"

    def on_key_down(self, window, key, *args):
        # Space key for screenshot
        if key == 32:
            self.take_screenshot()

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
        description="Run the Gelsight Mini Viewer with an optional config file."
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
    GelsightMini(config=gs_config.config).run()
