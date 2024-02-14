# customized dnn
import CNN_MNIST as dnn

# For image pixel
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors

# Additional libraries
import random
import time
from multiprocessing import Pool


# Return base_image
def build_base(meta_data, images):
    image_rows = []
    for i in range(meta_data["num_rows"]):
        start = meta_data["num_columns"] * i
        end = (meta_data["num_columns"]) * (i + 1)
        image_row = np.concatenate((images[start:end]), axis=-1)
        image_rows.append(image_row)
    image_grid = np.concatenate((image_rows), axis=2)
    base_image = np.squeeze(image_grid)
    return base_image


# Return mask image
def build_mask(meta_data, results):
    mask_cell = np.full((28, 28), np.nan)
    mask_rows = []
    current_idx = 0
    for row in range(meta_data["num_rows"]):
        mask_row = []
        for column in range(meta_data["num_columns"]):
            if current_idx in results:
                mask_cell = np.full((28, 28), results[current_idx])
            else:
                mask_cell = np.full((28, 28), np.nan)

            mask_row.append(mask_cell)
            current_idx += 1

        # Merge row of columns
        mask_row = np.array(mask_row)
        merged_mask_row = np.concatenate((mask_row), axis=-1)
        mask_rows.append(merged_mask_row)

    # Merge rows
    mask_image = np.concatenate((mask_rows), axis=0)
    return mask_image


# Refresh mask image
def animate_mask(meta_data, results):
    # Plot mask image
    mask_image = build_mask(meta_data, results)
    mask = plt.imshow(
        mask_image, cmap=meta_data["custom_cmap"], norm=meta_data["norm"], alpha=0.5
    )
    plt.pause(1e-10)
    return mask


# Return dict of dl and list of idx
def rand_batch(meta_data, dl, rand):
    # Batch dictionary
    batch_dict = {}
    batch_array = []
    for idx, batch in enumerate(dl):
        if idx > (meta_data["num_cells"]):
            break
        batch_dict[idx] = batch
        batch_array.append(batch)
    rand_idx = [*range(0, meta_data["num_cells"])]
    if rand:
        random.shuffle(rand_idx)
    return batch_dict, batch_array, rand_idx


# Initiate plt
def init_plt():
    # Initiate plt
    plt.subplots()
    # Configure plt
    plt.tight_layout(pad=1, w_pad=0, h_pad=0)
    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom", ["red", "white", "green"]
    )
    norm = plt.Normalize(-1, 1)
    return custom_cmap, norm


def initiate_model():
    meta_data = {}
    # Define batch_size
    meta_data["batch_size"] = 1
    meta_data["num_rows"] = 200
    meta_data["num_columns"] = 300
    meta_data["num_cells"] = meta_data["num_rows"] * meta_data["num_columns"]
    meta_data["threads"] = 4

    # init
    dl, images = dnn.dl_images()

    # Initiate plt
    custom_cmap, norm = init_plt()
    meta_data["custom_cmap"] = custom_cmap
    meta_data["norm"] = norm

    # Plot base image
    base_image = build_base(meta_data, images)
    base = plt.imshow(base_image, cmap="gray_r")

    # Initiate model
    model = dnn.init_model(meta_data["batch_size"])

    # Get dictonary of batches and random index
    batch_dict, batch_array, rand_idx = rand_batch(meta_data, dl, False)
    return meta_data, model, batch_dict, batch_array, rand_idx


# main only run if called directly (prevent multiprocessing crash)
if __name__ == "__main__":
    # initiate CNN model with MNIST images
    meta_data, model, batch_dict, batch_array, rand_idx = initiate_model()

    # Time before run
    start_time = time.time()

    results = {}

    # Call multiprocessing pool with set number of threads
    with Pool(meta_data["threads"]) as pool:
        # Make prediction using multiprocessing pool
        for result in pool.starmap(model.run, zip(rand_idx, batch_array)):
            results.update(result)

    # Time after run
    end_time = time.time()

    # Plot result in matplotlib
    animate_mask(meta_data, results)

    # Calculate Accuracy
    neg_list = list(results.values())
    count = 0
    # iterating each number in list
    for num in neg_list:
        # checking condition
        if num <= 0:
            count += 1

    # Print Logs
    print(f"Thread(s) used: {meta_data['threads']}")
    print(f"Total Item: {meta_data['num_cells']}")
    print(f"Total Runtime: {(end_time - start_time):.4f} sec")
    print(f"Accuracy: {((meta_data['num_cells']-count)/meta_data['num_cells'])*100:.2f}%")

    # Stay open
    plt.show()
