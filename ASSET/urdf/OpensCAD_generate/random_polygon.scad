
                        p1 = [0.024714122445207556, 0.005856151862934991];
                        p2 = [0.024497204174517076, 0.019948254100679342];
                        p3 = [-0.013153861690080176, 0.019840952092155226];
                        p4 = [-0.024550563711868964, 0.01715471649157126];
                        p5 = [-0.02415322817041728, 8.51811347042479e-05];
                        p6 = [-0.019911915049509128, -0.006912079789447412];
                        points = [p1, p2, p3, p4, p5, p6];
                        linear_extrude(height=0.01)
                        polygon(points);
                        