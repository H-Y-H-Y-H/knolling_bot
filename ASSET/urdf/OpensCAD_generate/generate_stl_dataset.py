from stl import mesh
import numpy as np
import copy
import os
import csv
import random

# Dictionary mapping categories to their max dimensions
category_dimensions = {
    'plier': (8.0, 4.0),  # length, width in cm
    'hammer': (8.0, 4.0),
    'chisel': (8.0, 4.0),
    'screwdriver': (8.0, 4.0),
    'wrench': (8.0, 4.0),
    'utilityknife': (8.0, 4.0),
    'gear': (5.0, 3.6),
    'usbdriver': (5.0, 3.6),
    'motor': (5.0, 3.6),
    'charger': (5.0, 3.6),
}

def calculate_centroid(stl_mesh):
    # Calculate the centroid of the STL object
    centroid = np.mean(stl_mesh.points.reshape(-1, 3), axis=0)
    return centroid

def translate_to_origin(stl_mesh, centroid):
    centroid_expanded = np.tile(centroid, (len(stl_mesh.points), 3))
    # Translate all vertices so the centroid is at the origin
    stl_mesh.points -= centroid_expanded
    return stl_mesh

def calculate_scale_factor(current_dimensions, max_dimensions):
    # Calculate scale factors for each dimension
    scale_factors = np.divide(max_dimensions, current_dimensions)
    # Return the smallest scale factor to maintain the aspect ratio
    return min(scale_factors)

def scale_mesh(original_mesh, max_dimensions):
    current_dimensions = calculate_3D_bounding_box(original_mesh)
    # Get the uniform scale factor to fit the object within the max dimensions
    uniform_scale_factor = calculate_scale_factor(current_dimensions[0:2], max_dimensions)
    # Scale the mesh
    original_mesh.vectors *= uniform_scale_factor
    return original_mesh
    
def calculate_3D_bounding_box(original_mesh):
    # Calculate the bounding box of the objecte
    min_values = np.min(original_mesh.vectors, axis=(0, 1))
    max_values = np.max(original_mesh.vectors, axis=(0, 1))
    current_dimensions = max_values - min_values
    return current_dimensions

def modify_and_save(original_mesh, scale_factor, variation_number, output_dir, file_prefix):
    object_type = file_prefix.split('_')[0]
    object_identifier = file_prefix.split('_')[1]
    parent_folder = os.path.join(output_dir, object_type)
    
    # Create the parent folder if it does not exist
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    # Now create a subfolder for the specific object variant inside the parent folder
    object_folder = os.path.join(parent_folder, f"{file_prefix}")
    if not os.path.exists(object_folder):
        os.makedirs(object_folder)
    
    file_name = os.path.join(object_folder, f"{file_prefix}_L{scale_factor[0]:.2f}_T{scale_factor[2]:.2f}")
    file_name_stl = f"{file_prefix}_L{scale_factor[0]:.2f}_T{scale_factor[2]:.2f}.stl"
    file_name_csv = f"{file_prefix}_L{scale_factor[0]:.2f}_T{scale_factor[2]:.2f}.csv"

    if not os.path.exists(file_name):
        os.makedirs(file_name)

    # Create a new mesh copy and apply the scale factor
    modified_mesh = copy.deepcopy(original_mesh)
    modified_mesh.vectors *= scale_factor
    # Construct the output file path
    output_file_path = os.path.join(file_name, file_name_stl)
    # Calculate the centroid and translate the mesh
    centroid = calculate_centroid(modified_mesh)
    modified_mesh = translate_to_origin(modified_mesh, centroid)
    # Save the modified STL with centroid aligned
    modified_mesh.save(output_file_path)


    bounding_box_dimensions = calculate_3D_bounding_box(modified_mesh)
    bounding_box_dimensions_list = bounding_box_dimensions.tolist()
    bounding_box_dimensions_list = [round(num,2) for num in bounding_box_dimensions_list]
    csv_file_path = os.path.join(file_name, file_name_csv)
    with open(csv_file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if csvfile.tell() == 0:
            csvwriter.writerow(["FileName", "LengthFactor", "ThicknessFactor", "BoundingBoxDimensions (cm)"])
        # Write the data
        csvwriter.writerow([file_name_csv, scale_factor[0], scale_factor[2], bounding_box_dimensions_list])

def process_stl_files(input_dir, output_dir):
    for category, max_dims in category_dimensions.items():
        category_path = os.path.join(input_dir, category)
        if os.path.exists(category_path):
            for file in os.listdir(category_path):
                if file.endswith('.stl'):
                    # Load the original STL file
                    file_path = os.path.join(category_path, file)
                    original_mesh = mesh.Mesh.from_file(file_path)
                    # Scale the mesh to fit the category's max dimensions
                    scaled_mesh = scale_mesh(original_mesh, max_dims)
                    # Generate variations with different sizes and thicknesses
                    for i in range(10):
                        # Introduce random scaling factors for X and Y axes
                        random_scale_x = 1 + random.uniform(-0.05, 0.05)  # Random scaling for X axis
                        random_scale_y = 1 + random.uniform(-0.05, 0.05)  # Random scaling for Y axis

                        thickness_factor = (1 - (i * 0.05)) * random_scale_y  # Adjusted thickness
                        length_factor = (1 - (i * 0.05)) * random_scale_x    # Adjusted length

                        modify_and_save(scaled_mesh, (length_factor, length_factor, thickness_factor), i, output_dir, file.split('.')[0])

def main():
    input_dir = 'electronics_stl'
    output_dir = 'generated_stl'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_stl_files(input_dir, output_dir)

if __name__ == "__main__":
    main()
