# Object Reconstruction Code Usage
The code for the reconstruction part is largely based on our last paper [NaturalTransRecon](https://github.com/arkgao/NaturalTransRecon). We use it to estimate the object masks, environment lighting and perform reconstruction. You may need to check our previou paper and code to better understand this program. And you can also use other methods to get masks, like the Grounded-SAM in recent work [TransparentGS](https://letianhuang.github.io/transparentgs/).

## Download Data
You can download our synthetic and real data from [Google Drive](https://drive.google.com/drive/folders/15g7lB6r1XMR8gNZi_8rZKjg2vkZpAdKy?usp=drive_link)
The data structure should be like:
```
CorresTrans/
|-- data/
    |-- case_name_1/
        |-- corresmap/          # the ground truth visualized correspondence for each view
        |-- image/              # multi-viwe image
        |-- mask/               # multi-view masks
        |-- normal/             # multi-view normal
        |-- cameras_sphere.npz  # camera params
        |-- object_sphere.npz   # camera params
        |-- gt.ply              # ground truth mesh
        |-- min_z.txt           # minimum z-coordinate of object, see below for explanation
    |-- case_name_2/
```

Our method only use image to reconstruct transparent objects, and the mask, normal, corresmap, out_dir in synthetic data are only used to validate results. For real data, there is no grounth truth data except mesh. We scan the painted real objects to get their GT shapes, while the pigment may introduce error and noise on ground truth.

We assume z-axis points towards the sky and the the base of the object is perpendicular to the z-axis. Since our data are all captured on the upper hemisphere of the object, the bottom surface of the object cannot be constrained by the mask. But the invisible bottom surface will still participate in the refraction and be optimized. Sometimes, it would protruding outward. So we need min_z as an extra constraint. We will release a guideline to explain how to prepare this on real data.

We collecte the open-source mesh files from [Wu et al.](https://vcc.tech/research/2018/FRT), [OmniObject3D](https://omniobject3d.github.io/) and [Poly Haven](https://polyhaven.com/), and place them in our data for the convenience of future users. The environment lighting we use is from the Lavel dataset and cannot be redistributed, so it is not included in the data.

After preparing the data, you can run python script step by step as follows:

## Stage1: Preparation
1. Frist, we use our previous method to get object masks and environment lighting. Since we remove the plane used in the previous paper, we made some modifications to the original code.
    ```shell
    python exp_runner.py --case CASE_NAME

    # export the multi-view object masks
    # for synthetic data
    python export_mask.py --case CASE_NAME
    
    # for real data
    python export_mask.py --case CASE_NAME --real_data
    ```
    The only difference is that it would perform extra open operation on mask to remove the noise caused by the upholder in real data.
    
2. Then we use the recoverd environment lighting to render object-free background images for each view.
    ```shell
    # For synthetic data
    python render_background.py --case CASE_NAME

    # For real data
    python render_backgroudn.py --case CASE_NAME --x_fov 90
    ```
    The only difference here is that we use the original fov to render the background image for synthetic data, while assign a fixed fov for real data to handle various real cases. For your custom data, we recommand use x_fov=90 (degree) as we used on real data.

3. Then we use pre-trained RCNet to estimate correspondence for each view
   ```shell
   python predict_correspondence.py --case CASE_NAME
   ```
   As mentioned above, we still working on organizing this part and the file is empty now. For now, you can put our results into the exp/case_name/pred_correspondence and run the following code. (Note, use our predicted results from [Google Drive](https://drive.google.com/drive/folders/15g7lB6r1XMR8gNZi_8rZKjg2vkZpAdKy?usp=drive_link) rather than the ground-truth in data. And the result file is a little large, approximately 300 M for each case. )



The results would be saved in exp folder, and it should be like:
```
RCTrans/
|-- exp/
    |-- case_name/
        |-- export_mask/
            |-- margin/
            |-- mask/
        |-- pred_correspondence
            |-- out_dir.npy
            |-- valid_mask.npy
        |-- render_background/
            |-- view/
                |-- background_0.png
                |-- ...
            |-- envmap.png
        |-- stage1/
```


## Stage2: Reconstruction
```shell
# initialize the object shape with masks
python init_shape.py --case case_name

# optimize shape through refractive ray tracing
python optim_transparent.py --case case_name    # add --val_error to calculate the error for syn data
#! For real data, there is misalignment and scalar ambiguity between GT and reconstruction
# Do not use this way to calculate error for real data
```
The final mesh is stored in exp/case_name/optim_trans/meshes/00300000.ply
```
TransRecon/
|-- exp/
    |-- case_name/
        |-- export_mask/
        |-- init_shape/
        |-- optim_trans/
        |-- pred_correspondence/
        |-- render_background/
        |-- stage1/
```
## Comparasion with color-supervision
In addition to optimizing the object using correspondence, this code also allows for optimization using rendered colors, which is modified version of [Gao et al. 2023] that we compared in our paper.
The previous steps are all the same. You just need to replace the last step with
```shell
python optim_transparent.py --case case_name --conf confs/optim_trans_withcolor.conf
```
It would render the object with coarse-to-fine blurred environment map and use the rendered color to optimize the object.

## Our results
For reference and comparasion, we provide all our results in [Google Drive](https://drive.google.com/drive/folders/15g7lB6r1XMR8gNZi_8rZKjg2vkZpAdKy?usp=drive_link).
