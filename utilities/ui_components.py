# ui_common.py
import os
import threading
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.modalview import ModalView
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from kivy.uix.button import Button
from kivy.metrics import dp
from kivy.app import App


# --- Loading Overlay ---
class ConnectingOverlay(ModalView):
    def __init__(self, message="Connecting...", **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0, 0, 0, 1)
        self.auto_dismiss = False
        self.size_hint = (1, 1)
        self.add_widget(Label(text=message, font_size="22sp", color=(1, 1, 1, 1)))


# --- FileChooser Popup using default elements ---
class FileChooserPopup(Popup):
    def __init__(self, select_callback, **kwargs):
        super().__init__(title="Select Folder", size_hint=(0.8, 0.8), **kwargs)
        self.select_callback = select_callback  # Callback to handle the selection
        layout = BoxLayout(orientation="vertical")
        self.filechooser = FileChooserListView(path="/", dirselect=True)
        layout.add_widget(self.filechooser)
        # Button layout (Confirm + Cancel)
        button_layout = BoxLayout(size_hint_y=None, height=dp(50))
        select_button = Button(
            text="Select this Folder", size_hint=(None, None), size=(dp(180), dp(30))
        )
        select_button.bind(on_release=self.select_folder)
        cancel_button = Button(
            text="Cancel", size_hint=(None, None), size=(dp(180), dp(30))
        )
        cancel_button.bind(on_release=self.dismiss)
        button_layout.add_widget(select_button)
        button_layout.add_widget(cancel_button)
        layout.add_widget(button_layout)
        self.add_widget(layout)

    def select_folder(self, instance):
        if self.filechooser.selection:
            self.select_callback(self.filechooser.selection[0])
        self.dismiss()


# --- Top Bar with Device Spinner ---
class TopBar(BoxLayout):
    def __init__(self, on_device_selected_callback, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.size_hint_y = None
        self.height = dp(50)
        self.padding = [dp(10)] * 4

        left = BoxLayout(orientation="horizontal")
        left.add_widget(
            Label(
                text="Selected device:",
                size_hint=(None, 1),
                width=dp(120),
                color=(1, 1, 1, 1),
            )
        )
        from utilities.gelsightmini import (
            Camera,
        )  # import here to avoid circular dependency if needed

        available_devices = Camera.list_devices()
        spinner_values = [f"Device {k}" for k in sorted(available_devices.keys())]
        self.device_spinner = Spinner(
            text="Select Device",
            values=spinner_values,
            size_hint=(None, 1),
            width=dp(120),
        )
        self.device_spinner.bind(
            text=lambda spinner, text: self.on_device_selected(
                text, on_device_selected_callback
            )
        )
        left.add_widget(self.device_spinner)
        self.add_widget(left)

    def on_device_selected(self, text, callback):
        try:
            device_index = int(text.split()[-1])
        except Exception:
            device_index = 0
        app = App.get_running_app()
        app.show_overlay("Connecting...")
        threading.Thread(target=callback, args=(device_index,), daemon=True).start()
