from __future__ import annotations
from os.path import join as pathjoin
from time import time
from typing import Tuple, Union
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

import drjit as dr
import mitsuba as mi

scene_dir = pathjoin("scenes", "mitsuba3_tutorial_scenes")
gt_dir = output_dir = pathjoin(".", "gt_rendered")
scene_labels = ["cbox", "cbox_point"]
gt_spp = 40960
# gt_file = "cbox_40960spp.npy"

def imshow_compare(img, ref, title_img="My answer", title_ref="GT",
                   opt_img=dict(), opt_ref=dict(), opt_diff=dict()):
    plt.subplot(131)
    plt.imshow(scale_gamma(img), **opt_img)
    plt.title(title_img)
    
    plt.subplot(132)
    plt.imshow(scale_gamma(ref), **opt_ref)
    plt.title(title_ref)
    
    plt.subplot(133)
    im = plt.imshow((img - ref).sum(-1), cmap='seismic', **opt_diff)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Diff.")

def text_at(scene: mi.Scene, point: Union[mi.Point3f, ArrayLike],
            text: str, ha: str="center", va: str="top"):
    pixpos = world2img(scene, point)
    xoffset = 0
    yoffset = 10 if va == "top" else -10
    kargs = dict(size=10, ha=ha, va=va, bbox=dict(ec = (0.,0.,0.,0.),
                                                  fc=(1.,1.,1., 0.3)))
    plt.text(pixpos[0] + xoffset, pixpos[1] + yoffset,
             text, **kargs)
    
def test_integrator_short(integrator_name):
    # Options
    max_depth = 20
    spp = 1024
    gamma = 1/2.2
    print_header = "[test_integrator_short] "
    # print_indent = " " * len(print_header)
    gt_path = pathjoin(gt_dir, gt_file)
    
    # Main
    scene = mi.load_dict(mi.cornell_box())
    integrator = mi.load_dict({'type': integrator_name, 'max_depth': max_depth})
    t0 = time()
    image = mi.render(scene, integrator=integrator, spp=spp).numpy()
    print(f"{print_header}Rendering {spp} spp using {integrator_name} integrator done in {time()-t0:.2f} sec.")
    
    image_gt = np.load(gt_path)
    
    plt.subplot(131)
    plt.imshow(image ** gamma)
    plt.title(f"{integrator_name}, {spp} spp")
    
    plt.subplot(132)
    plt.imshow(image_gt ** gamma)
    plt.title(f"GT")
    
    plt.subplot(133)
    im = plt.imshow((image - image_gt).sum(-1))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"Rendering error")
    
    plt.show()
    
def test_sequence(func_seq: callable[int, mi.Scene], scene_limit: mi.scene,
                  title_seq: str, title_limit: str,
                  opt_seq=dict(), opt_limit=dict(), opt_diff=dict()):
    '''
    title_seq should contain a formatting character
    '''
    #--- Sequence configs
    n_list = [1, 5, 25]
    spp = 256

    #--- Main
    num_col = len(n_list) + 1
    img_limit =  mi.render(scene_limit, spp=spp).numpy()
    plt.subplot(2, num_col, num_col)
    plt.imshow(scale_gamma(img_limit), **opt_limit)
    plt.title(title_limit)
    
    for i, n in enumerate(n_list):
        img_seq = mi.render(func_seq(n), spp=spp).numpy()
        plt.subplot(2, num_col, i+1)
        plt.imshow(scale_gamma(img_seq), **opt_seq)
        plt.title(title_seq % n)

        plt.subplot(2, num_col, i+1+num_col)
        im = plt.imshow((img_seq - img_limit).sum(-1), cmap='seismic', **opt_diff)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Diff. from target limit")
    
    plt.show()

def world2img(scene: mi.Scene, point: Union[mi.Point3f, ArrayLike]) -> mi.Point2u:
    '''
    Assume height and width are equal.
    '''
    params = mi.traverse(scene)
    sensor_tf = params['sensor.to_world']
    size = params['sensor.film.size']
    w, h = size
    x_fov = params['sensor.x_fov']
    xhalf = dr.tan(dr.deg2rad(x_fov/2))
    yhalf = xhalf * h / w
    
    point_in_sensor = sensor_tf.inverse() @ mi.Point3f(point)
    point_imgplane = -mi.Point2f(point_in_sensor.x/point_in_sensor.z, point_in_sensor.y/point_in_sensor.z)
    pix_mi = mi.Point2u((point_imgplane / mi.Point2f(xhalf, yhalf) + 1) * size / 2) - params['sensor.film.crop_offset']
    return pix_mi

def primary_rays(scene: mi.Scene) -> mi.Ray3f:
    sensor = scene.sensors()[0]
    
    # Modified from: Lines 316-000 in https://github.com/mitsuba-renderer/mitsuba3/blob/master/src/python/python/ad/integrators/common.py
    film = sensor.film()
    film_size = film.crop_size()
    rfilter = film.rfilter()
    border_size = rfilter.border_size()

    if film.sample_border():
        film_size += 2 * border_size

    # Compute discrete sample position
    idx = dr.arange(mi.UInt32, dr.prod(film_size))

    # Compute the position on the image plane
    pos = mi.Vector2i()
    pos.y = idx // film_size[0]
    pos.x = dr.fma(-film_size[0], pos.y, idx)

    if film.sample_border():
        pos -= border_size

    pos += mi.Vector2i(film.crop_offset())

    # Cast to floating point and add random offset
    pos_f = mi.Vector2f(pos) + 0.5

    # Re-scale the position to [0, 1]^2
    scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
    offset = -mi.ScalarVector2f(film.crop_offset()) * scale
    pos_adjusted = dr.fma(pos_f, scale, offset)

    aperture_sample = mi.Vector2f(0.0)
    ray, _ = sensor.sample_ray(
        time=0,
        sample1=0,
        sample2=pos_adjusted,
        sample3=aperture_sample
    )
    return ray

# primary surface interactions
def primary_sis(scene: mi.Scene) -> mi.SurfaceInteraction3f:
    rays = primary_rays(scene)
    si = scene.ray_intersect(rays)
    return si

def primary_hits(scene: mi.Scene) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rays = primary_rays(scene)
    si = scene.ray_intersect(rays)
    
    pt = si.p.numpy()
    print(10, pt.shape)
    mask = si.is_valid().numpy()
    # prem_id = si.prim_index.numpy()
    shape_id = dr.full(mi.Int, -1)
    for i, shape in enumerate(scene.shapes()):
        shape_id[si.shape.eq_(mi.ShapePtr(shape))] = i
    shape_id = shape_id.numpy()
    
    sensor = scene.sensors()[0]
    film = sensor.film()
    w, h = film.crop_size()
    
    pt = pt.reshape(h, w, 3)
    mask = mask.reshape(h, w)
    # return pt.reshape(h, w, 3), mask.reshape(h, w), prem_id.reshape(h, w), shape_id.reshape(h, w)
    return pt.reshape(h, w, 3), mask.reshape(h, w), shape_id.reshape(h, w)

def scale_gamma(img: ArrayLike, gm: float=2.2) -> ArrayLike:
    # Clip to avoid warning message from `plt.imshow`
    img = np.clip(img, 0.0, 1.0)
    return img ** (1/gm)

def scale_vec2unit(vec: ArrayLike) -> ArrayLike:
    # Clip to avoid warning message from `plt.imshow`
    unit_range = vec / 2 + 0.5
    return np.clip(unit_range, 0.0, 1.0)