import numpy as np
from scipy.spatial import ConvexHull
import subprocess
import os
from urdfpy import URDF
import shutil

def polygon_generate(path, start_evaluation, end_evaluation):
    # Generate OpenSCAD code dynamically
    num_vertices_max = 40
    x_range = np.array([-0.025, 0.025])
    y_range = np.array([-0.02, 0.02])


    total_point = []
    for i in range(start_evaluation, end_evaluation):
        num_vertices = 6
        x_data = np.random.uniform(x_range[0], x_range[1], num_vertices_max)
        y_data = np.random.uniform(y_range[0], y_range[1], num_vertices_max)
        points = np.concatenate((x_data.reshape(num_vertices_max, 1), y_data.reshape(num_vertices_max, 1)), axis=1)
        hull = ConvexHull(points)
        convex_points = points[hull.vertices[:num_vertices]]
        total_point = np.append(total_point, convex_points).reshape(-1, num_vertices * 2)
        openscad_code = f"""
                        p1 = [{convex_points[0, 0]}, {convex_points[0, 1]}];
                        p2 = [{convex_points[1, 0]}, {convex_points[1, 1]}];
                        p3 = [{convex_points[2, 0]}, {convex_points[2, 1]}];
                        p4 = [{convex_points[3, 0]}, {convex_points[3, 1]}];
                        p5 = [{convex_points[4, 0]}, {convex_points[4, 1]}];
                        p6 = [{convex_points[5, 0]}, {convex_points[5, 1]}];
                        points = [p1, p2, p3, p4, p5, p6];
                        linear_extrude(height=0.01)
                        polygon(points);
                        """
        with open("random_polygon.scad", "w") as file:
            file.write(openscad_code)
        output_file = path + f"polygon_{i}.stl"
        print(f'this is num{i}')
        command = ['openscad', '-o', output_file, '--export-format=binstl', 'random_polygon.scad']
        subprocess.run(command)
    np.savetxt(path + 'points_%s_%s.txt' % (start_evaluation, end_evaluation), total_point)

def box_generate():
    data = np.load('data.txt', allow_pickle=True)

    # Read the text file and parse cube sizes
    with open("data.txt", "r") as file:
        cube_sizes = [list(map(float, line.strip().split())) for line in file]

    # Generate OpenSCAD code dynamically
    openscad_code = ""
    for i, size in enumerate(cube_sizes):
        size = np.round(size[2:4], decimals=3)
        length = size[0] * 1000
        width = size[1] * 1000
        height = 12
        print(int(length), int(width), int(height))
        openscad_code = f"""
            cube([{int(length)}, {int(width)}, {int(height)}], center = true);
        """

        # Save the generated OpenSCAD code to a file
        with open("batch.scad", "w") as file:
            file.write(openscad_code)

        # Execute OpenSCAD command to generate STL files
        output_file = f"cube_{int(length)}_{int(width)}_{int(height)}.stl"
        print(f'this is num{i}, size is {size}')
        subprocess.run(["C:/Program Files (x86)/openscad/openscad.exe", "-o", output_file, "batch.scad"])

def sundry_generate(path, start_evaluation, end_evaluation):

    nut_dimension_metric = 5
    wrench_thick = 10
    wrench_length = 40
    wrench_code = f"""
                        //******************************************
                        //* Base configuration
                        //******************************************
                        
                        //Set the nut properties
                        NUT_DIMENSION_METRIC = /*M*/{nut_dimension_metric}; // this is the number after M (eg. M4 -> 4)
                        // >>> NOTE <<<
                        // the metric parameter works ONLY for nuts from M1.6 up to M20
                        // If your nut has different size, please set it here:
                        // (leave -1 if you use metric)
                        NUT_DIMENSION = -1/*mm*/;
                        
                        WRENCH_THICKNESS_MM         = {wrench_thick};    // this is the thickness of the wrench   
                        WRENCH_LENGHT_MM            = {wrench_length};   // this is the lenght of the wrench
                        CORRECT_LENGTH_IF_TOO_SHORT = true; // correct wrench lenght if is too short according with nut size
                        
                        // Choose if you want both closed and open heads, or only one of them
                        ADD_OPEN_HEAD = true;
                        ADD_CLOSED_HEAD = true;
                        
                        //-------------------------------------------------------
                        // PRINT HINTS
                        // Higher the infill rate is, stronger the wrench will be
                        // I suggest, at least, a 30% honeycomb fill
                        //-------------------------------------------------------
                        
                        
                        
                        //******************************************
                        //* Advanced configuration
                        //******************************************
                        // Change thickness of the border around nut hole
                        // A thicker border make wrench stronger, but wider
                        //  - Positive value: additional thickness around nut hole
                        //  - Negative value: less thickness around nut hole
                        fHeadBoardersAdditionalThickness = 0;
                        
                        // According to your printer precision, you can change 
                        // the 'nut holes offset' value. This affect the size of nut hole.
                        // Positive value: bigger hole
                        // Negative value: smaller hole
                        fNutHoleOffet = 0.2;
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        //********************************************************************************
                        //* Do not modify below this line
                        //********************************************************************************
                        
                        // Variables cannot be assigned with if statement, so do this trick
                        iNutDiameter = (NUT_DIMENSION > 0) ? (NUT_DIMENSION) : (
                            (NUT_DIMENSION_METRIC == 1.6) ? 3.2 : (
                                (NUT_DIMENSION_METRIC == 2) ? 4 : (
                                    (NUT_DIMENSION_METRIC == 2.5) ? 5 : (
                                        (NUT_DIMENSION_METRIC == 3) ? 5.5 : (
                                            (NUT_DIMENSION_METRIC == 4) ? 7 : (
                                                (NUT_DIMENSION_METRIC == 5) ? 8 : (
                                                    (NUT_DIMENSION_METRIC == 6) ? 10 : (
                                                        (NUT_DIMENSION_METRIC == 8) ? 13 : (
                                                            (NUT_DIMENSION_METRIC == 10) ? 16 : (
                                                                (NUT_DIMENSION_METRIC == 12) ? 18 : (
                                                                    (NUT_DIMENSION_METRIC == 14) ? 21 : (
                                                                        (NUT_DIMENSION_METRIC == 16) ? 24 : (
                                                                            (NUT_DIMENSION_METRIC == 20) ? 30 : (
                                                                                // For nut > M20 or not standard return 0
                                                                                0
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        );
                        
                        iNutThickness = WRENCH_THICKNESS_MM;
                        
                        // Calculate the radius of cylinder which model the nut 
                        // (it will be a 6-faces cylinder, so an hexagon)
                        iNutCylinderRadius = iNutDiameter/2 + fNutHoleOffet;
                        
                        // Calculate the radius of the circular part around the
                        // nut hole.
                        
                        
                        // Calculate the header space around nut hole using
                        // linear function
                        iHeadSpaceForM2 = min(2, fHeadBoardersAdditionalThickness);
                        iHeadSpaceForM10 = 7 + fHeadBoardersAdditionalThickness;  // 13mm of diameter
                        
                        iM2Diameter = 4;
                        iM10Diameter = 13;
                        // two points are:
                        // A = (iM2Diameter, iHeadSpaceForM2) 
                        // B = (iM10Diameter, iHeadSpaceForM10)
                        //              yB - yA
                        //    y - yA = -------- (x - xA)
                        //              xB - xA
                        iHeaderRadiusOffset = ((iHeadSpaceForM10-iHeadSpaceForM2) / (iM10Diameter-iM2Diameter)) * (iNutDiameter - iM2Diameter) + iHeadSpaceForM2;
                        
                        //iHeaderRadiusOffset = 0.225*iNutDiameter + 0.75;
                        
                        echo("Nut diameter is:", iNutDiameter);
                        echo("Offset is:", iHeaderRadiusOffset);
                        iHeadRadius = iNutCylinderRadius + iHeaderRadiusOffset;
                        
                        /*
                         * See this page:
                         * https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/undersized_circular_objects
                         */
                        module cylinder_outer(height,radius,fn)
                        {{
                           fudge = 1/cos(180/fn);
                           cylinder(h=height,r=radius*fudge,$fn=fn);
                        }}
                        
                        module createNutProfile()
                        {{
                            // Add a little bit of offset on the height and traslate down
                            // of the same amount, to avoid wrong booleand operation with
                            // the rounded section
                            iNutHeightOffset = 0.5;
                            translate([0,0,-iNutHeightOffset/2]){{
                                rotate(a=[0,0,90]){{
                                    // The nut profile is obtained from a cylinder with 6 faces
                                    cylinder_outer(iNutThickness+iNutHeightOffset, iNutCylinderRadius, 6);
                                }}
                            }}
                        }}
                        
                        module createNutAdditionalSquaredHole(iPosY)
                        {{
                            iSize = iNutCylinderRadius*2;
                            iHeightOffset = 0.2;
                            iHeight = WRENCH_THICKNESS_MM + iHeightOffset;
                            translate([0,iPosY,iHeight/2 - iHeightOffset/2]){{
                                cube(size = [iSize, iSize, iHeight], center = true);
                            }};
                        }}
                        
                        module createCylindricalHead()
                        {{
                            rotate(a=[0,0,90]){{
                                cylinder_outer(iNutThickness, iHeadRadius, 50);
                            }}
                        }}
                        
                        module createOpenHead(iPosY=0)
                        {{
                            translate([0,iPosY,0]){{
                                /*
                                 * The nut hexagon is created via 'cylinder outer' so
                                 * the cylinder diameter is the length of face-to-face segment
                                 * and NOT the angle-to-angle segment.
                                 * So we need to calculate it
                                 * According to https://en.wikipedia.org/wiki/Hexagon we calculate
                                 * the value of 'R'based on 'r'
                                 */    
                                iNutAngleToAngleLenght = iNutCylinderRadius/ cos(30);    
                                // Do the same for the cylinder around the nut hole
                                iHeadAngleToAngleLenght = iHeadRadius/ cos(30);
                                
                                iAdjustNutPos = iNutAngleToAngleLenght/3;
                                
                                iDelta = (iHeadAngleToAngleLenght - iNutAngleToAngleLenght) - iAdjustNutPos - fHeadBoardersAdditionalThickness;
                               
                                difference(){{
                                    createCylindricalHead();
                                    translate([0, -(iDelta), 0 / 2]) {{
                                        createNutProfile();
                                    }};
                                    createNutAdditionalSquaredHole(-(iDelta+iNutAngleToAngleLenght));
                                }}
                                
                            }}
                            
                        }}
                        
                        module createClosedHead(iPosY=0)
                        {{
                            translate([0,iPosY,0]){{
                                difference(){{
                                    createCylindricalHead();
                                    createNutProfile();
                                }}
                            }}
                        }}
                        
                        module createRoundedHandleEdge(iPosY, iHandleWidth)
                        {{
                            iRadius = iHandleWidth/2;
                            translate([0,iPosY,0]){{
                                cylinder(h=WRENCH_THICKNESS_MM,r=iRadius,$fn=20);
                            }}
                        }}
                        
                        module create3DText(sText, iSize, iThickness)
                        {{
                            font = "Arial";
                            //letter_size = 50;
                            //letter_height = 1;
                            // Use linear_extrude() to make the letters 3D objects as they
                            // are only 2D shapes when only using text()
                            
                            linear_extrude(height = iThickness) {{
                                text(sText, size = iSize, font = font, halign = "center", valign = "center", $fn = 16);
                            }}
                        }}
                        
                        module addTextToHandle(sText, iPosY, iTextSize)
                        {{
                            iTextThickness = 0.4;
                            
                            iTextSize = min(iTextSize, 5);
                            
                            iTextYOffsetToBooleanSubtraction = 0.1;
                            translate([0, iPosY, WRENCH_THICKNESS_MM - iTextYOffsetToBooleanSubtraction]){{
                                rotate([0,0,90]){{
                                    create3DText(sText, iTextSize - 1, iTextThickness + iTextYOffsetToBooleanSubtraction);
                                }};
                            }};
                        }}
                        
                        iMinDist = 10;
                        function calculateCorrectedLenght() = ((WRENCH_LENGHT_MM - (iNutDiameter*2) > iMinDist) ? WRENCH_LENGHT_MM : (iNutDiameter*2 + iMinDist));
                        
                        function getWrenchLenght() = ( (CORRECT_LENGTH_IF_TOO_SHORT == true) ? calculateCorrectedLenght() : WRENCH_LENGHT_MM );
                        
                        module createHandle()
                        {{
                            // Calculate if the length user set
                            iUserLenght = getWrenchLenght();
                            
                            iLength = iUserLenght - (iNutCylinderRadius*2 + 0.5);
                            iWidth = iNutCylinderRadius*1.6;
                            
                            sText = (NUT_DIMENSION > 0) ? (str(NUT_DIMENSION, "mm")) : (str("M", NUT_DIMENSION_METRIC));
                            
                            union(){{
                                
                                difference(){{
                                    // Create the handle with a parallelepiped
                                    translate([0,0,(WRENCH_THICKNESS_MM/2)]){{
                                        cube(
                                            size = [
                                                iWidth,
                                                iLength,
                                                WRENCH_THICKNESS_MM
                                            ], 
                                            center = true
                                        );
                                    }};
                                    // Subtract the text
                                    //addTextToHandle(sText, (iLength/2) - 5, iWidth);
                                    addTextToHandle(sText, 0, iWidth);
                                }} // close difference()
                                
                                // Put a rounded edge if not both head exist
                                if( ADD_OPEN_HEAD == false){{
                                    createRoundedHandleEdge(-(iLength/2), iWidth);
                                }}
                                if(ADD_CLOSED_HEAD == false){{
                                    createRoundedHandleEdge(iLength/2, iWidth);
                                }}
                            }} // close union()
                            
                        }}
                        
                        
                        
                        
                        
                        
                        
                        // Create the handle section
                        createHandle();
                        // then closed and open section
                        iDistanceFromOrigin = getWrenchLenght()/2;
                        if( ADD_OPEN_HEAD == true){{
                            createOpenHead(-iDistanceFromOrigin);
                        }}
                        if( ADD_CLOSED_HEAD == true){{
                            createClosedHead(iDistanceFromOrigin);
                        }}
                    """
    with open("wrench.scad", "w") as file:
        file.write(wrench_code)
    output_file = path + f"wrench_{0}.stl"
    print(f'this is num{0}')
    command = ['openscad', '-o', output_file, '--export-format=binstl', 'wrench.scad']
    subprocess.run(command)

def stl2urdf(start, end, tar_path):

    # total_data = np.loadtxt(tar_path + 'points_%s_%s.txt' % (start, end))
    #
    # for i in range(start, end):
    #     points_data = total_data[i - start]
    #     # temp = URDF.load('../OpensCAD_generate/template.urdf')
    #     # temp.name = 'polygon_%s' % i
    #     # temp.base_link.visuals[0].geometry.mesh.filename = 'polygon_%s.stl' % i
    #     # temp.base_link.collisions[0].geometry.mesh.filename = 'polygon_%s.stl' % i
    #     # temp.save(tar_path + 'polygon_%s.urdf' % i)
    #     shutil.copy('template.urdf', tar_path + 'polygon_%s.urdf' % i)
    #     pass
    # for i in range(start, end):
    #     with open(tar_path + 'polygon_%s.urdf' % i, "r") as file:
    #         data = file.read()
    #         new_data = data.replace('polygon_0.stl', 'polygon_%d.stl' % i)
    #     with open(tar_path + 'polygon_%s.urdf' % i, "w") as file:
    #         file.write(new_data)

    shutil.copy('template.urdf', tar_path + 'wrench_0.urdf')

    with open(tar_path + 'wrench_0.urdf', "r") as file:
        data = file.read()
        new_data = data.replace('polygon_0.stl', 'wrench_0.stl')
    with open(tar_path + 'wrench_0.urdf', "w") as file:
        file.write(new_data)

def read_stl_vertice(path):

    from stl import mesh

    stl_mesh = mesh.Mesh.from_file(path + 'wrench_0.stl')

    pass

if __name__ == '__main__':

    # path = '../../../../knolling_dataset/random_polygon/'
    # os.makedirs(path, exist_ok=True)
    # start_evaluation = 400
    # end_evaluation = 600
    # polygon_generate(path, start_evaluation, end_evaluation)
    #
    # tar_path = '../../../../knolling_dataset/random_polygon/'
    # os.makedirs(tar_path, exist_ok=True)
    # start = 400
    # end = 600

    # box_generate()

    path = '../sundry/'
    tar_path = '../sundry/'
    os.makedirs(path, exist_ok=True)
    start_evaluation = 400
    end_evaluation = 600
    # sundry_generate(path, start_evaluation, end_evaluation)

    # stl2urdf(0, 0, tar_path)

    # tar_path = '../../../../knolling_dataset/random_polygon/'
    read_stl_vertice(path)
