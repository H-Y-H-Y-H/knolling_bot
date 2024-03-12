import os

def generate_urdf(stl_file, urdf_directory):
    object_name = os.path.splitext(os.path.basename(stl_file))[0]
    urdf_content = f"""<?xml version="1.0" encoding="utf-8"?>
<robot name="{object_name}">
  <link name="{object_name}Link">
    <contact>
      <lateral_friction value="1"/>
    </contact>
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.1"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{stl_file}" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{stl_file}" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
  </link>
</robot>"""

    urdf_file_path = os.path.join(urdf_directory, f"{object_name}.urdf")
    with open(urdf_file_path, 'w') as urdf_file:
        urdf_file.write(urdf_content)

    print(f"Generated URDF for {object_name}: {urdf_file_path}")

def main():
    stl_directory = 'generated_stl'
    urdf_directory = 'urdf_file'
    os.makedirs(urdf_directory, exist_ok=True)

    for root, dirs, files in os.walk(stl_directory):
        for filename in files:
            if filename.endswith('.stl'):
                stl_file_path = os.path.join(root, filename)
                generate_urdf(stl_file_path, urdf_directory)

if __name__ == '__main__':
    main()
