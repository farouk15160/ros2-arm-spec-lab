"""Optional MuJoCo RGB-D rendering; all output depth uses metres along optical Z."""
import numpy as np


class CameraRenderer:
    """Owned GL resources: create and use from one thread, call close on shutdown."""

    def __init__(self, model, cameras, seed=0):
        import mujoco
        self._model = model
        self._cameras = {camera['name']: camera for camera in cameras}
        # MuJoCo stores shared visual clipping as multiples of scene extent.
        # Convert explicit SI camera ranges before allocating any GL context.
        if self._cameras:
            extent = float(model.stat.extent)
            if not np.isfinite(extent) or extent <= 0:
                raise ValueError('camera rendering requires positive finite scene extent')
            model.vis.map.znear = min(camera['clip'][0] for camera in self._cameras.values()) / extent
            model.vis.map.zfar = max(camera['clip'][1] for camera in self._cameras.values()) / extent
        self._rng = np.random.default_rng(seed)
        self._renderers = {}
        try:
            for name, camera in self._cameras.items():
                width, height = camera['resolution']
                self._renderers[name] = mujoco.Renderer(model, height=height, width=width)
        except Exception:
            self.close()
            raise

    def render(self, data, name):
        camera, renderer = self._cameras[name], self._renderers[name]
        renderer.update_scene(data, camera=name)
        result = {'frame_id': camera['optical_frame'], 'stamp': float(data.time),
                  'intrinsics': dict(camera['intrinsics'])}
        if camera['type'] in ('rgb', 'rgbd'):
            renderer.disable_depth_rendering()
            result = {**result, 'rgb': renderer.render().copy()}
        if camera['type'] in ('depth', 'rgbd'):
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            near, far = camera['clip']
            render_far = float(self._model.vis.map.zfar * self._model.stat.extent)
            valid = np.isfinite(depth) & (depth >= near) & (depth < min(far, render_far * (1-1e-5)))
            if camera['noise_stddev']:
                depth = depth + self._rng.normal(0., camera['noise_stddev'], depth.shape)
            valid = valid & np.isfinite(depth) & (depth >= near) & (depth < far)
            result = {**result, 'depth': np.where(valid, depth, np.nan).astype(np.float32)}
            renderer.disable_depth_rendering()
        return result

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers = {}
