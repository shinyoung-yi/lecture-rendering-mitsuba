from __future__ import annotations
from os.path import join as pathjoin
from time import time
from typing import Tuple, Union, Sequence, Optional
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

def normalize(vec:  ArrayLike,
              axis: Optional[int] = -1
             ) ->   ArrayLike:
    return vec / np.linalg.norm(vec, axis=axis, keepdims=True)

def sph2vec(theta: ArrayLike, # [*]
            phi:   ArrayLike, # [*]
            axis:  Optional[int] = -1
           ) ->    ArrayLike: # [*, 3] if `axis == -1`
    sint = np.sin(theta)
    x = sint*np.cos(phi)
    y = sint*np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=axis)

def vec2sph(vec:  ArrayLike, # *x3 (default), not assumed normalized
            axis: int = -1,
           ) ->   Tuple[ArrayLike, ArrayLike]: # (theta[*], phi[*]) in radian
    vec = np.asarray(vec)
    if vec.shape[axis] != 3:
        raise ValueError(f"The size along given axis({axis}) should be equal to 3, but curretly shape:{vec.shape}.")
    vec = normalize(vec, axis=axis)
    vec = np.swapaxes(vec, axis, -1)
    X = vec[...,0]
    Y = vec[...,1]
    Z = vec[...,2]
    theta = np.arccos(Z)
    phi = np.arctan2(Y, X)
    return theta, phi

def scale_gamma(img: ArrayLike, gm: float=2.2) -> ArrayLike:
    # Clip to avoid warning message from `plt.imshow`
    img = np.clip(img, 0.0, 1.0)
    return img ** (1/gm)

def scale_vec2unit(vec: ArrayLike) -> ArrayLike:
    # Clip to avoid warning message from `plt.imshow`
    unit_range = vec / 2 + 0.5
    return np.clip(unit_range, 0.0, 1.0)

def imshow_compare(img, ref, title_img="My answer", title_ref="GT",
                   opt_img=dict(), opt_ref=dict(), opt_diff=dict()):
    img = np.asarray(img)
    plt.subplot(131)
    plt.imshow(scale_gamma(img), **opt_img)
    plt.title(title_img)
    
    if type(ref) == str:
        ref = mi.Bitmap(ref)
    ref = np.asarray(ref)
    plt.subplot(132)
    plt.imshow(scale_gamma(ref), **opt_ref)
    plt.title(title_ref)
    
    plt.subplot(133)
    diff = (img - ref).sum(-1)
    vabs = max(np.abs(diff.max()), np.abs(diff.min()))
    im = plt.imshow(diff, cmap='seismic', vmin=-vabs, vmax=vabs, **opt_diff)
    
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Diff.")
    plt.tight_layout()

def imshow_compare_many(img_list: Sequence[ArrayLike],
                        ref: Union[ArrayLike, str],
                        title_img_list: Optional[Sequence[str]] = None,
                        title_ref: Optional[str] = "GT",
                        vabs: Optional[float] = None,
                        opt_img: dict = dict(),
                        opt_ref: dict = dict(),
                        opt_diff: dict = dict()):
    n_img = len(img_list)
    if type(ref) == str:
        ref = mi.Bitmap(ref)
    ref = np.asarray(ref)

    for i, img in enumerate(img_list):
        img = np.asarray(img)
        plt.subplot(2, n_img+1, i+1)
        plt.imshow(scale_gamma(img), **opt_img)
        
        if title_img_list is None:
            title_img = f"My answer {i+1}"
        else:
            title_img = title_img_list[i]
        plt.title(title_img)

        plt.subplot(2, n_img+1, i+n_img+2)
        diff = (img - ref).sum(-1)
        if vabs is None:
            vabs_curr = max(np.abs(diff.max()), np.abs(diff.min()))
        else:
            vabs_curr = vabs
        im = plt.imshow(diff, cmap='seismic', vmin=-vabs_curr, vmax=vabs_curr, **opt_diff)

        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Diff.")
    
    plt.subplot(2, n_img+1, n_img+1)
    plt.imshow(scale_gamma(ref), **opt_ref)
    plt.title(title_ref)
    plt.tight_layout()


def text_at(scene: mi.Scene, point: Union[mi.Point3f, ArrayLike],
            text: str, ha: str="center", va: str="top"):
    pixpos = np.squeeze(world2img(scene, point).numpy())
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
    if not isinstance(scene, mi.Scene):
        raise TypeError(f"Wrong type for `scene`: {type(scene)}")
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

##################################################
### Mitsuba scene dictionary
##################################################

# "Mi"tsuba objects in "dict"ionary
def midict_area(rgb: ArrayLike # [3]
               ) ->  dict:
    return {'type': 'area',
            'radiance': {'type': 'rgb',
                         'value': rgb}}

def midict_diffuse(reflectance: ArrayLike) -> dict:
    return {'type': 'diffuse',
            'reflectance': {'type': 'rgb',
                            'value': reflectance}}

def midict_sphere(center:  ArrayLike,
                  radius:  float,
                  bsdf:    Union[dict, str],
                  emitter: Union[dict, str, None] = None
                 ) ->      dict:
    res = {'type': 'sphere',
           'center': center,
           'radius': radius}
    if type(bsdf) == str:
        res['bsdf'] = {'type': 'ref', 'id': bsdf}
    elif type(bsdf) == dict:
        res['bsdf'] = bsdf
    else:
        raise TypeError(f"Unsupported type ({type(bsdf)}) for `bsdf`.")
    if type(emitter) == str:
        res['emitter'] = {'type': 'ref', 'id': emitter}
    elif type(emitter) == dict:
        res['emitter'] = emitter
    elif emitter is not None:
        raise TypeError(f"Unsupported type ({type(emitter)}) for `emitter`.")
    return res

def scale_scene_dict(scene_dict: dict, factor: float) -> dict:
    T = mi.ScalarTransform4f
    res_dict = {}
    for k, v in scene_dict.items():
        if type(v) == dict:
            v_res = v.copy()
        else:
            v_res = v
        if k == 'sensor':
            tf = v['to_world']
            npmat = tf.matrix.numpy()
            # print(type(npmat))
            # print(npmat)
            # print(k)
            # print(k*npmat[:3, 3])
            npmat[:3, 3] *= factor
            v_res['to_world'] = mi.ScalarTransform4f(npmat)
        else:
            if 'to_world' in v:
                tf = v['to_world']
                v_res['to_world'] = T.scale([factor, factor, factor]) @ tf
            for mult_able in ['position', 'center', 'radius']:
                if mult_able in v:
                    v_res[mult_able] = v[mult_able] * factor
        res_dict[k] = v_res
    return res_dict

##################################################
### matplotlib.pyplot
##################################################

def text_at(point: mi.Point3f, scene: mi.Scene, text: str, ha: str="center", va: str="top"):
    pixpos = world2img(scene, point)
    xoffset = 0
    yoffset = 10 if va == "top" else -10
    kargs = dict(size=10, ha=ha, va=va, bbox=dict(ec = (0.,0.,0.,0.),
                                                  fc=(1.,1.,1., 0.3)))
    plt.text(pixpos[0] + xoffset, pixpos[1] + yoffset,
             text, **kargs)

def show_ray(scene_dict: dict, ray: mi.Ray3f, si: mi.SurfaceInteraction3f):
    scene_dict['zero_bsdf'] = midict_diffuse([0, 0, 0])
    
    scene_dict['ray'] = {
        'type': 'cylinder',
        'p0': ray.o,
        'p1': si.p,
        'radius': 0.01,
        'bsdf': {'type': 'ref', 'id': 'zero_bsdf'},
        'emitter': midict_area([1, 0, 0])}
    scene_dict['origin'] = midict_sphere(ray.o, 0.03, 'zero_bsdf', midict_area([1, 1, 1]))
    scene_dict['intersection'] = midict_sphere(si.p, 0.03, 'zero_bsdf', midict_area([2, 0.4, 0.4]))

    scene_visualize = mi.load_dict(scene_dict)
    img = mi.render(scene_visualize, spp=100)
    plt.imshow(img**(1/2.2))
    
    with np.printoptions(precision=4):
        text_at(ray.o, scene_visualize, f"ray.o = mi.Point3f({ray.o.numpy()})")
        text_at(si.p, scene_visualize, f"si.p = mi.Point3f({si.p.numpy()})", ha="right", va="bottom")
    
def show_ds(si: mi.SurfaceInteraction3f, ds_list: Sequence[mi.DirectionSample3f]):
    scene_dict = mi.cornell_box()
    scene_dict['integrator']['max_depth'] = 5
    scene_dict['zero_bsdf'] = midict_diffuse([0, 0, 0])

    # For visibility of `ds` on the light source
    scene_dict['light']['emitter']['radiance']['value'] = [0.2, 0.2, 0.2]
    scene_dict['point'] = {'type': 'point',
                           'position': [0, 0.2, 0.3],
                           'intensity': {'type': 'rgb', 'value': [0.3,0.3,0.3]}} 
    
    scene_dict['si'] = midict_sphere(si.p, 0.03, 'zero_bsdf', midict_area([0.4, 0.4, 2]))
    for i, ds in enumerate(ds_list):
        scene_dict[f'ds{i}'] = midict_sphere(ds.p, 0.005, 'zero_bsdf', midict_area([2, 0.4, 0.4]))
    scene_visualize = mi.load_dict(scene_dict)
    spp = 512
    img = mi.render(scene_visualize, spp=spp)
    plt.subplot(1, 2, 1)
    plt.imshow(img**(1/2.2))

    text_at(si.p, scene_visualize, f"si.p = mi.Point3f({si.p.numpy().round(2)})", ha='right', va='top')
    # ds0p = ds_list[0].p
    # text_at(ds0p, scene_visualize, f"ds_list[0].p = mi.Point3f({ds0p})", ha="center", va="top")

    scene_dict['sensor']['to_world'] = mi.ScalarTransform4f.look_at([0, 0, 0], [0, 1, 0], [0, 0, 1])
    scene_visualize = mi.load_dict(scene_dict)
    img = mi.render(scene_visualize, spp=spp)
    plt.subplot(1, 2, 2)
    plt.imshow(img**(1/2.2))