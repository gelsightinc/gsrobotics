import cv2
import numpy as np
from config import ConfigModel
from utilities.image_processing import (
    stack_label_above_image,
    apply_cmap,
    color_map_from_txt,
    normalize_array,
    trim_outliers,
)
from utilities.reconstruction import Reconstruction3D
from utilities.visualization import Visualize3D
from utilities.gelsightmini import GelSightMini
from utilities.logger import log_message


def UpdateView(
    image: np.ndarray,
    cam_stream: GelSightMini,
    reconstruction: Reconstruction3D,
    visualizer3D: Visualize3D,
    cmap: np.ndarray,
    config: ConfigModel,
    window_title: str,
):

    # Compute depth map and gradients.
    depth_map, contact_mask, grad_x, grad_y = reconstruction.get_depthmap(
        image=image,
        markers_threshold=(config.marker_mask_min, config.marker_mask_max),
    )

    if visualizer3D:
        visualizer3D.update(depth_map, gradient_x=grad_x, gradient_y=grad_y)

    if np.isnan(depth_map).any():
        return

    depth_map_trimmed = trim_outliers(depth_map, 1, 99)
    depth_map_normalized = normalize_array(array=depth_map_trimmed, min_divider=10)
    depth_rgb = apply_cmap(data=depth_map_normalized, cmap=cmap)

    # Convert masks to 8-bit grayscale.
    contact_mask = (contact_mask * 255).astype(np.uint8)

    # Convert grayscale images to 3-channel for stacking.
    contact_mask_rgb = cv2.cvtColor(contact_mask, cv2.COLOR_GRAY2BGR)

    # Apply labels above images
    frame_labeled = stack_label_above_image(
        image, f"Camera Feed {int(cam_stream.fps)} FPS", 30
    )

    depth_labeled = stack_label_above_image(depth_rgb, "Depth", 30)
    contact_mask_labeled = stack_label_above_image(contact_mask_rgb, "Contact Mask", 30)
    # Increase spacing between images by adding black spacers
    spacing_size = 30
    horizontal_spacer = np.zeros(
        (frame_labeled.shape[0], spacing_size, 3), dtype=np.uint8
    )

    top_row = np.hstack(
        (
            frame_labeled,
            horizontal_spacer,
            contact_mask_labeled,
            horizontal_spacer,
            depth_labeled,
        )
    )

    display_frame = top_row

    # Scale the display frame
    display_frame = cv2.resize(
        display_frame,
        (
            int(display_frame.shape[1] * config.cv_image_stack_scale),
            int(display_frame.shape[0] * config.cv_image_stack_scale),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    display_frame = display_frame.astype(np.uint8)

    # Show the combined image.
    cv2.imshow(window_title, display_frame)


def View3D(config: ConfigModel):
    WINDOW_TITLE = "Multi-View (Camera, Contact, Depth)"

    reconstruction = Reconstruction3D(
        image_width=config.camera_width,
        image_height=config.camera_height,
        use_gpu=config.use_gpu,  # Change to True if you want to use CUDA.
    )

    # Load the trained network using the existing method in reconstruction.py.
    if reconstruction.load_nn(config.nn_model_path) is None:
        log_message("Failed to load model. Exiting.")
        return

    if config.pointcloud_enabled:
        # Initialize the 3D Visualizer.
        visualizer3D = Visualize3D(
            pointcloud_size_x=config.camera_width,
            pointcloud_size_y=config.camera_height,
            save_path="",  # Provide a path if you want to save point clouds.
            window_width=int(config.pointcloud_window_scale * config.camera_width),
            window_height=int(config.pointcloud_window_scale * config.camera_height),
        )
    else:
        visualizer3D = None

    cmap = color_map_from_txt(
        path=config.cmap_txt_path, is_bgr=config.cmap_in_BGR_format
    )

    # Initialize the camera stream.
    cam_stream = GelSightMini(
        target_width=config.camera_width, target_height=config.camera_height
    )
    devices = cam_stream.get_device_list()
    log_message(f"Available camera devices: {devices}")
    # For testing, select device index 0 (adjust if needed).
    cam_stream.select_device(config.default_camera_index)
    cam_stream.start()

    # Main loop: capture frames, compute depth map, and update the 3D view.
    try:
        while True:
            # Get a new frame from the camera.
            frame = cam_stream.update(dt=0)
            if frame is None:
                continue

            # Convert color
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            UpdateView(
                image=frame,
                cam_stream=cam_stream,
                reconstruction=reconstruction,
                visualizer3D=visualizer3D,
                cmap=cmap,
                config=config,
                window_title=WINDOW_TITLE,
            )

            # Exit conditions.
            # When press 'q' on keyboard
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Check if the window has been closed by the user.
            # cv2.getWindowProperty returns a value < 1 when the window is closed.
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                # workaround to better catch widnow exit request
                for _ in range(5):
                    cv2.waitKey(1)
                break

    except KeyboardInterrupt:
        log_message("Exiting...")
    finally:
        # Release the camera and close windows.
        if cam_stream.camera is not None:
            cam_stream.camera.release()
        cv2.destroyAllWindows()
        if visualizer3D:
            visualizer3D.visualizer.destroy_window()


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
    View3D(config=gs_config.config)
