import numpy as np
import open3d as o3d
import cv2 as cv
import os
from joblib import Parallel, delayed
import tqdm


def gen_bev_map(pc, colors=None, lr_range=[-20, 20], bf_range=[-20, 20], res=0.1):
    pc = pc.reshape(-1, 3)

    # sort by hight test change
    arr1inds = pc[..., 2].argsort()
    # pc = pc[arr1inds[::-1]]
    # colors = colors[arr1inds[::-1]]
    pc = pc[arr1inds]
    if not isinstance(colors, type(None)):
        colors = colors.reshape(-1, 3)
        colors = colors[arr1inds]

    x = pc[..., 0].flatten()
    y = pc[..., 1].flatten()
    z = pc[..., 2].flatten()

    # filter point cloud
    f_filt = np.logical_and((x > bf_range[0]), (x < bf_range[1]))
    s_filt = np.logical_and((y > -lr_range[1]), (y < -lr_range[0]))
    filt = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filt).flatten()
    x = x[indices]
    y = y[indices]
    z = z[indices]

    # Festlegen der Bildgröße auf 512x512
    # w, h = 512, 512
    # convert coordinates to
    x_img = (-y/res).astype(np.int32)
    y_img = (-x/res).astype(np.int32)
    # shifting image, make min pixel is 0,0
    x_img -= int(np.floor(lr_range[0]/res))
    y_img += int(np.ceil(bf_range[1]/res))

    # pixel_values = color
    # according to width and height generate image
    w = 1+int((lr_range[1] - lr_range[0])/res)
    h = 1+int((bf_range[1] - bf_range[0])/res)

    # crop y to make it not bigger than 255
    height_range = (-2, 2.0)
    pixel_values = np.clip(a=z, a_min=height_range[0], a_max=height_range[1])

    def scale_to_255(a, min, max, dtype=np.uint8):
        return (((a - min) / float(max - min)) * 255).astype(dtype)
    if isinstance(colors, type(None)):
        pixel_values = scale_to_255(
            pixel_values, min=height_range[0], max=height_range[1])
        im = np.zeros([h, w], dtype=np.uint8)
    else:
        colors = colors.reshape((-1, 3))
        colors = colors[indices]
        pixel_values = colors
        im = np.zeros([h, w, 3])  # , dtype=np.uint8)

    im[y_img, x_img] = pixel_values

    return im


def is_road_present(image, 
                    road_color=np.array([255, 0, 255], dtype=np.uint8), 
                    min_road_pixels=100,
                    roi_width=200,
                    roi_height=200):
    """
    ueberprueft, ob eine durchgehende Linie von mindestens 70 road_color-Pixeln vorhanden ist // wtf? why?
    und ob mindestens 20 % der Pixel im Bild road_color sind.
    """
    # we just create a roi in the middle of the image and check if there's a certain amount of road colored pixels
    # this even works for non pink road
    road_mask = np.all(image == road_color, axis=-1)
    roi_x = (int(road_mask.shape[0]/2 - roi_width/2), int(road_mask.shape[0]/2 + roi_width/2))
    roi_y = (int(road_mask.shape[1]/2 - roi_height/2), int(road_mask.shape[1]/2 + roi_height/2))
    road_mask_roi = road_mask[roi_x[0]:roi_y[1], roi_y[0]:roi_y[1]]
    if np.where(road_mask_roi)[0].shape[0] >= min_road_pixels:
        print("found road")
        return True
    else:
        print("no road found")
        return False


def _has_min_length_line(mask, min_length):
    # this might be one of the worst ideas i have ever seen. even worse than oli's approach to search for specific scenes
    """
    Hilfsfunktion, um zu ueberpruefen, ob eine Linie von mindestens min_length in einer Maske vorhanden ist.
    """
    # ueberpruefe jede Zeile und jede Spalte
    for axis in [0, 1]:
        if np.any([np.sum(line) >= min_length for line in np.apply_along_axis(np.convolve, axis, mask, np.ones(min_length), mode='valid')]):
            return True

    # ueberpruefe Diagonalen # omg? really? why?
    for k in range(-mask.shape[0]+min_length, mask.shape[1]-min_length):
        if np.sum(np.diag(mask, k=k)) >= min_length or np.sum(np.diag(np.fliplr(mask), k=k)) >= min_length:
            return True

    return False


def create_bev(path, x, y, window_size, output_heights_folder, output_normals_folder, output_bev_folder, num, i, j):
    pc = o3d.io.read_point_cloud(path)
    bev_output = (gen_bev_map(np.asarray(pc.points), np.asarray(pc.colors), lr_range=[int(y), int(
            y+window_size)], bf_range=[int(x), int(x+window_size)], res=0.1) * 255).astype(np.uint8)
    
    if np.any(bev_output != 0) :
        if is_road_present(bev_output):
            bev_normals = gen_bev_map(np.asarray(pc.points), 0.5*(np.asarray(pc.normals)+1), lr_range=[
                                    int(y), int(y+window_size)], bf_range=[int(x), int(x+window_size)], res=0.1)*255
            bev = gen_bev_map(np.asarray(pc.points), None, lr_range=[int(y), int(
            y+window_size)], bf_range=[int(x), int(x+window_size)], res=0.1).astype(np.uint8)*255
            
            cv.imwrite(os.path.join(output_heights_folder,
                        f'{num}_row{i}_col{j}_bev_height.png'), bev)
            cv.imwrite(os.path.join(output_normals_folder,
                        f'{num}_row{i}_col{j}_bev_normals.png'), bev_normals)
            cv.imwrite(os.path.join(output_bev_folder,
                        f'{num}_row{i}_col{j}_bev_output_color.png'), bev_output)


def main():

    for num in tqdm.tqdm(range(0, 10), desc="sequence"):
        count = 0
        path = os.path.expanduser(
            '~') + f"/data/christian/kitti_sequence_{num:02d}/kitti_{num:02d}_vdbmesh.ply"
        output_path = os.path.expanduser('~') + "/data/christian/bevcc"
        output_heights_folder = os.path.join(output_path, "heights")
        output_normals_folder = os.path.join(output_path, "normals")
        output_bev_folder = os.path.join(output_path, "bev")
        os.makedirs(output_heights_folder, exist_ok=True)
        os.makedirs(output_normals_folder, exist_ok=True)
        os.makedirs(output_bev_folder, exist_ok=True)

        pc = o3d.io.read_point_cloud(path)
        pc_array = np.asarray(pc.points)
        # Definieren Sie minx, maxx, miny, maxy hier innerhalb der main-Funktion
        minx, maxx = np.min(pc_array[:, 0]) + 20, np.max(pc_array[:, 0]) - 20
        miny, maxy = np.min(pc_array[:, 1]) + 20, np.max(pc_array[:, 1]) - 20

        window_size = 40
        overlap_percentage = 0.70  # 70% Overlapping
        # Schrittgröße mit Overlapping
        step_size = window_size * (1 - overlap_percentage)

        for i, x in enumerate(tqdm.tqdm(np.arange(minx, maxx, step_size), desc="x")):
            #for j, y in enumerate(np.arange(miny, maxy, step_size)):
                #create_bev(path, x, y, window_size, output_heights_folder, output_normals_folder, output_bev_folder, num, i, j)
            Parallel(n_jobs=10)(delayed(create_bev)(path, x, y, window_size, output_heights_folder, output_normals_folder, output_bev_folder, num, i, j)
                                for j, y in enumerate(np.arange(miny, maxy, step_size)))


if __name__ == "__main__":
    main()
