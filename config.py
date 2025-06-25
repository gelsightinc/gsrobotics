import json
import os
from pathlib import Path
from pydantic import BaseModel, ValidationError
from utilities.logger import log_message


class ConfigModel(BaseModel):
    default_camera_index: int
    camera_width: int
    camera_height: int
    second_camera_width: int
    second_camera_height: int
    border_fraction: float
    marker_mask_min: int
    marker_mask_max: int
    pointcloud_enabled: bool
    pointcloud_window_scale: float
    cv_image_stack_scale: float
    nn_model_path: str
    cmap_txt_path: str
    cmap_in_BGR_format: bool
    use_gpu: bool


default_config = ConfigModel(
    default_camera_index=0,
    camera_width=320,
    camera_height=240,
    second_camera_width=320,
    second_camera_height=240,
    border_fraction=0.15,
    marker_mask_min=0,
    marker_mask_max=70,
    pointcloud_enabled=True,
    pointcloud_window_scale=3,
    cv_image_stack_scale=1.5,
    nn_model_path="./models/nnmini.pt",
    cmap_txt_path="./cmap.txt",
    cmap_in_BGR_format=True,
    use_gpu=False,
)


def get_absolute_path(config_path: str) -> Path:
    """
    Returns an absolute Path object, for cross-platform compativility

    Args:
        config_path (str): provided string path

    Returns:
        Path: absolute path
    """
    return Path(config_path).expanduser().resolve()


class GSConfig:
    def __init__(self, config_path: str = None):
        self.config_path = get_absolute_path(config_path) if config_path else None
        self.config: ConfigModel = default_config

        if self.config_path and self.config_path.is_file():
            self.load_config()

        else:
            log_message(f"No path for config provided. Using default configuration.")

    def load_config(self):
        """
        Loads configuration from the specifiied JSON file, and ensures that this file
        matches desired structure. If it is invalid, default config is used instead.
        """
        if self.config_path is None:
            self.config = default_config
            log_message(f"No config path. Using default configuration.")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            self.config = ConfigModel(**data)
            log_message(f"Loaded config data from file: {self.config_path}.")

        except (json.JSONDecoder, ValidationError, KeyError) as e:
            log_message(
                f"Warning: Invalid config file. Using default configuration. Error: {e}"
            )
            self.config = default_config

    def save_config(self, save_path: str = None):
        """
        Saves current configuration to a JSON file.

        Args:
            save_path (str, optional): Path for saving the config. Defaults to None.
        """

        path = get_absolute_path(save_path) if save_path else self.config_path

        if not path:
            log_message("No valid path provided for saving configuration. Skipping...")
            return

        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.config.model_dump(), file, indent=4)
        log_message(f"Configuration saved to {path}")

    def reset_to_default(self):
        """
        Resets the current config to the default one.
        """

        self.config = default_config.model_copy(deep=True)
        log_message("Configuration reset to default.")


if __name__ == "__main__":
    # create instance of gs config
    gs_config = GSConfig()
    # when no path privided, default config is loaded
    gs_config.load_config()
    # saving default convig to json in the root directory
    gs_config.save_config("default_config.json")
