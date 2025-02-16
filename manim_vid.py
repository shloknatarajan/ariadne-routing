from manim import *

class PromptToVector(MovingCameraScene):
    def construct(self):
        # Step 1: Write out the prompt.
        start_text = Text("Let's start with our query").scale(0.7).to_edge(DOWN)
        prompt = Text("Query: Propose a benefits package for an entry-level SWE").scale(0.7)
        self.play(Write(start_text))
        self.wait(0.5)
        self.play(Write(prompt))
        self.play(FadeOut(start_text))
        self.wait(1)
        # self.play(FadeOut(start_text))

        # Step 2: Create a 2D grid and transform the text into a vector.
        grid = NumberPlane()
        grid.set_opacity(0.5)
        self.play(Create(grid))
        vector_text = Text("The prompt becomes a point in vector space").scale(0.7).to_edge(DOWN)
        self.play(Write(vector_text))
        self.wait(0.5)

        # Create a vector arrow and its label.
        vector_arrow = Arrow(start=ORIGIN, end=2*RIGHT + 2*UP, buff=0, color=GREEN)
        vector_label = MathTex(r"\vec{v}").next_to(vector_arrow.get_end(), RIGHT)
        vector = VGroup(vector_arrow, vector_label)
        self.play(ReplacementTransform(prompt, vector))

        # Create text that will appear later
        
        self.wait(1)
        self.play(FadeOut(vector_text))
        

        # Create a green dot at the arrow's endpoint
        endpoint_dot = Dot(point=vector_arrow.get_end(), color=GREEN)
        
        vector_text = Text("And it'll be quite close to similar prompts").scale(0.7).to_edge(DOWN)
        self.play(Write(vector_text))

        # Make the vector arrow fade out and show the dot
        self.play(FadeOut(vector_arrow), FadeIn(endpoint_dot), FadeOut(vector_text))
        self.wait(0.5)

        # Step 3: Add other points near the vector (a cluster).
        
        
        dot1 = Dot(point=vector_arrow.get_end() + UP*0.5 + RIGHT*0.3, color=BLUE)
        dot2 = Dot(point=vector_arrow.get_end() + DOWN*0.4 + LEFT*0.2, color=BLUE)
        dot3 = Dot(point=vector_arrow.get_end() + RIGHT*0.5 + DOWN*0.3, color=BLUE)
        dot4 = Dot(point=vector_arrow.get_end() + LEFT*0.4 + UP*0.3, color=BLUE)
        self.play(FadeIn(dot1), FadeIn(dot2), FadeIn(dot3), FadeIn(dot4))
        # Group the vector and dots together.
        cluster = VGroup(vector_label, dot1, dot2, dot3, dot4, endpoint_dot)
        
        
        
        self.wait(1)
        # self.play(FadeOut(vector_text))
        vector_text = Text("If we zoom out, we'll see a few other clusters").scale(0.7).to_edge(DOWN)
        self.play(Write(vector_text))
        # Step 4: Zoom out and show multiple clusters
        self.play(
            self.camera.frame.animate.scale(2),
            
        )
        self.wait(0.5)
        
        
        # Double the density by halving the grid spacing
        # Create a denser grid with half the spacing
        new_grid = NumberPlane(
            x_range=(-2*7.111111111111111, 2*7.111111111111111, 1), # Half the step size
            y_range=(-2*4.0, 2*4.0, 1), # Half the step size
            x_length=2*14.22222222222222, # Original x range width
            y_length=2*8.0 # Original y range height
        )
        new_grid.set_opacity(0.5)
        self.play(
            ReplacementTransform(grid, new_grid, run_time=2, rate_func=smooth)
        )
        self.play(FadeOut(vector_text))
        # Draw circle around original cluster
        circ1 = Circle(color=GREEN, radius=1.2).surround(cluster)
        
        # Create additional clusters with different colors
        # Cluster 2 - Purple
        cluster2_center = 3*LEFT + 2*DOWN
        dots2 = VGroup(*[Dot(point=cluster2_center + np.array([np.random.uniform(-0.4, 0.4), 
                                                              np.random.uniform(-0.4, 0.4), 0]), 
                            color=PURPLE) for _ in range(5)])
        circ2 = Circle(color=PURPLE, radius=1).move_to(cluster2_center)
        
        # Cluster 3 - Orange 
        cluster3_center = 4*LEFT + 3*UP
        dots3 = VGroup(*[Dot(point=cluster3_center + np.array([np.random.uniform(-0.3, 0.3),
                                                              np.random.uniform(-0.3, 0.3), 0]),
                            color=ORANGE) for _ in range(4)])
        circ3 = Circle(color=ORANGE, radius=0.8).move_to(cluster3_center)
        
        # Cluster 4 - Pink
        cluster4_center = 3*RIGHT + 2*DOWN  
        dots4 = VGroup(*[Dot(point=cluster4_center + np.array([np.random.uniform(-0.5, 0.5),
                                                              np.random.uniform(-0.5, 0.5), 0]),
                            color=PINK) for _ in range(6)])
        circ4 = Circle(color=PINK, radius=1.3).move_to(cluster4_center)
        
        # Create all clusters and circles
        self.play(
            Create(circ1),
            FadeIn(dots2), Create(circ2),
            FadeIn(dots3), Create(circ3), 
            FadeIn(dots4), Create(circ4)
        )


        vector_text = Text("We now re-embed clusters to shift to 'Actor Space'").scale(1.1).set_y(-5)
        self.play(Write(vector_text))
        self.wait(1)

        self.play(FadeOut(vector_text))
        # Step 5: Move all clusters to new non-overlapping positions
        self.play(
            cluster.animate.shift(RIGHT*-3 + UP*2),
            circ1.animate.shift(RIGHT*-3 + UP*2),
            dots2.animate.shift(LEFT*4 + DOWN*3),
            circ2.animate.shift(LEFT*4 + DOWN*3),
            dots3.animate.shift(LEFT*-5 + UP*-3),
            circ3.animate.shift(LEFT*-5 + UP*-3),
            dots4.animate.shift(RIGHT*4 + DOWN*-2),
            circ4.animate.shift(RIGHT*4 + DOWN*-2)
        )
        self.wait(1)

        # Step 6: Add a red dot nearby the cluster.
        # Here, we position it a bit to the right of the cluster's center
        # Create multiple red stars at random positions
        stars = VGroup(*[
            Star(color=RED, fill_opacity=1).scale(0.2).move_to(
                np.array([
                    np.random.uniform(-6, 6),
                    np.random.uniform(-4, 4),
                    0
                ])
            ) for _ in range(8)
        ])
        
        vector_text = Text("And finally, we can map to our agents").scale(1.1).set_y(-5)
        self.play(Write(vector_text))
        self.wait(1)

        # Fade in the stars
        self.play(FadeIn(stars))
        self.wait(0.5)
        
        # Get cluster centers after movement
        cluster_centers = [
            cluster.get_center(),  # This one already accounts for movement
            cluster2_center + (LEFT*4 + DOWN*3),
            cluster3_center + (LEFT*-5 + UP*-3),
            cluster4_center + (RIGHT*4 + DOWN*-2)
        ]
        
        # Create lines between clusters and stars
        all_lines = []
        for center in cluster_centers:
            # Calculate distances to all stars
            distances = [(star, np.linalg.norm(center - star.get_center())) 
                        for star in stars]
            distances.sort(key=lambda x: x[1])
            
            # Create lines with decreasing opacity for 1st, 2nd, 3rd closest
            for i, (star, _) in enumerate(distances[:3]):
                opacity = 1.0 if i == 0 else (0.5 if i == 1 else 0.2)
                line = Line(
                    start=center,
                    end=star.get_center(),
                    stroke_opacity=opacity,
                    color=WHITE
                )
                all_lines.append(line)
        
        # Create all lines with a staggered animation
        self.play(
            *[Create(line, run_time=1.5) for line in all_lines],
            lag_ratio=0.1
        )
        self.wait(2)