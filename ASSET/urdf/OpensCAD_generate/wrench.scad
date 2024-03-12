
                        //******************************************
                        //* Base configuration
                        //******************************************
                        
                        //Set the nut properties
                        NUT_DIMENSION_METRIC = /*M*/5; // this is the number after M (eg. M4 -> 4)
                        // >>> NOTE <<<
                        // the metric parameter works ONLY for nuts from M1.6 up to M20
                        // If your nut has different size, please set it here:
                        // (leave -1 if you use metric)
                        NUT_DIMENSION = -1/*mm*/;
                        
                        WRENCH_THICKNESS_MM         = 10;    // this is the thickness of the wrench   
                        WRENCH_LENGHT_MM            = 40;   // this is the lenght of the wrench
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
                        {
                           fudge = 1/cos(180/fn);
                           cylinder(h=height,r=radius*fudge,$fn=fn);
                        }
                        
                        module createNutProfile()
                        {
                            // Add a little bit of offset on the height and traslate down
                            // of the same amount, to avoid wrong booleand operation with
                            // the rounded section
                            iNutHeightOffset = 0.5;
                            translate([0,0,-iNutHeightOffset/2]){
                                rotate(a=[0,0,90]){
                                    // The nut profile is obtained from a cylinder with 6 faces
                                    cylinder_outer(iNutThickness+iNutHeightOffset, iNutCylinderRadius, 6);
                                }
                            }
                        }
                        
                        module createNutAdditionalSquaredHole(iPosY)
                        {
                            iSize = iNutCylinderRadius*2;
                            iHeightOffset = 0.2;
                            iHeight = WRENCH_THICKNESS_MM + iHeightOffset;
                            translate([0,iPosY,iHeight/2 - iHeightOffset/2]){
                                cube(size = [iSize, iSize, iHeight], center = true);
                            };
                        }
                        
                        module createCylindricalHead()
                        {
                            rotate(a=[0,0,90]){
                                cylinder_outer(iNutThickness, iHeadRadius, 50);
                            }
                        }
                        
                        module createOpenHead(iPosY=0)
                        {
                            translate([0,iPosY,0]){
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
                               
                                difference(){
                                    createCylindricalHead();
                                    translate([0, -(iDelta), 0 / 2]) {
                                        createNutProfile();
                                    };
                                    createNutAdditionalSquaredHole(-(iDelta+iNutAngleToAngleLenght));
                                }
                                
                            }
                            
                        }
                        
                        module createClosedHead(iPosY=0)
                        {
                            translate([0,iPosY,0]){
                                difference(){
                                    createCylindricalHead();
                                    createNutProfile();
                                }
                            }
                        }
                        
                        module createRoundedHandleEdge(iPosY, iHandleWidth)
                        {
                            iRadius = iHandleWidth/2;
                            translate([0,iPosY,0]){
                                cylinder(h=WRENCH_THICKNESS_MM,r=iRadius,$fn=20);
                            }
                        }
                        
                        module create3DText(sText, iSize, iThickness)
                        {
                            font = "Arial";
                            //letter_size = 50;
                            //letter_height = 1;
                            // Use linear_extrude() to make the letters 3D objects as they
                            // are only 2D shapes when only using text()
                            
                            linear_extrude(height = iThickness) {
                                text(sText, size = iSize, font = font, halign = "center", valign = "center", $fn = 16);
                            }
                        }
                        
                        module addTextToHandle(sText, iPosY, iTextSize)
                        {
                            iTextThickness = 0.4;
                            
                            iTextSize = min(iTextSize, 5);
                            
                            iTextYOffsetToBooleanSubtraction = 0.1;
                            translate([0, iPosY, WRENCH_THICKNESS_MM - iTextYOffsetToBooleanSubtraction]){
                                rotate([0,0,90]){
                                    create3DText(sText, iTextSize - 1, iTextThickness + iTextYOffsetToBooleanSubtraction);
                                };
                            };
                        }
                        
                        iMinDist = 10;
                        function calculateCorrectedLenght() = ((WRENCH_LENGHT_MM - (iNutDiameter*2) > iMinDist) ? WRENCH_LENGHT_MM : (iNutDiameter*2 + iMinDist));
                        
                        function getWrenchLenght() = ( (CORRECT_LENGTH_IF_TOO_SHORT == true) ? calculateCorrectedLenght() : WRENCH_LENGHT_MM );
                        
                        module createHandle()
                        {
                            // Calculate if the length user set
                            iUserLenght = getWrenchLenght();
                            
                            iLength = iUserLenght - (iNutCylinderRadius*2 + 0.5);
                            iWidth = iNutCylinderRadius*1.6;
                            
                            sText = (NUT_DIMENSION > 0) ? (str(NUT_DIMENSION, "mm")) : (str("M", NUT_DIMENSION_METRIC));
                            
                            union(){
                                
                                difference(){
                                    // Create the handle with a parallelepiped
                                    translate([0,0,(WRENCH_THICKNESS_MM/2)]){
                                        cube(
                                            size = [
                                                iWidth,
                                                iLength,
                                                WRENCH_THICKNESS_MM
                                            ], 
                                            center = true
                                        );
                                    };
                                    // Subtract the text
                                    //addTextToHandle(sText, (iLength/2) - 5, iWidth);
                                    addTextToHandle(sText, 0, iWidth);
                                } // close difference()
                                
                                // Put a rounded edge if not both head exist
                                if( ADD_OPEN_HEAD == false){
                                    createRoundedHandleEdge(-(iLength/2), iWidth);
                                }
                                if(ADD_CLOSED_HEAD == false){
                                    createRoundedHandleEdge(iLength/2, iWidth);
                                }
                            } // close union()
                            
                        }
                        
                        
                        
                        
                        
                        
                        
                        // Create the handle section
                        createHandle();
                        // then closed and open section
                        iDistanceFromOrigin = getWrenchLenght()/2;
                        if( ADD_OPEN_HEAD == true){
                            createOpenHead(-iDistanceFromOrigin);
                        }
                        if( ADD_CLOSED_HEAD == true){
                            createClosedHead(iDistanceFromOrigin);
                        }
                    