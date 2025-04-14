# main.py
import numpy as np
import pyaudio
import curses
import time
import importlib
import pkgutil
import os
import inspect

# Import the base visualizer
from visualizer_base import VisualizerBase

# Import the visualizers package
import visualizers

class TerminalAudioVisualizer:
    def __init__(self):
        # Audio setup
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.pause = False
        
        # FFT and visualization parameters
        self.spectrum = np.zeros(self.CHUNK)
        self.smoothed_spectrum = np.zeros(self.CHUNK // 2)
        self.previous_spectrum = np.zeros(self.CHUNK // 2)
        self.smoothing = 0.8  # Smoothing factor
        self.energy = 0  # Current energy level
        
        # Sensitivity control
        self.sensitivity = 1.0  # Default sensitivity multiplier
        self.sensitivity_step = 0.1  # How much to change per keystroke
        
        # Visual effects
        self.hue_offset = 0
        
        # Load visualizers
        self.visualizers = self.load_visualizers()
        self.current_visualizer_index = 0
        
        # Initialize audio stream
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=False,
            frames_per_buffer=self.CHUNK
        )
    
    def load_visualizers(self):
        """Dynamically load all visualizer plugins"""
        visualizers_list = []
        
        # Iterate through all modules in the visualizers package
        for _, name, _ in pkgutil.iter_modules(visualizers.__path__):
            try:
                # Import the module
                module = importlib.import_module(f"visualizers.{name}")
                
                # Find all classes in the module that inherit from VisualizerBase
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, VisualizerBase) and obj is not VisualizerBase:
                        # Create an instance of the visualizer and add it to the list
                        visualizers_list.append(obj())
                        print(f"Loaded visualizer: {obj().name}")
            except Exception as e:
                print(f"Error loading visualizer {name}: {e}")
        
        if not visualizers_list:
            print("No visualizers found!")
            
        return visualizers_list
    
    def get_audio_data(self):
        # Read audio data
        data = np.frombuffer(self.stream.read(self.CHUNK, exception_on_overflow=False), dtype=np.int16)
        
        # Apply FFT to get frequency domain
        spectrum = np.abs(np.fft.fft(data)[:self.CHUNK // 2])
        
        # Normalize and apply smoothing
        spectrum = spectrum / (128 * self.CHUNK)
        self.previous_spectrum = self.smoothed_spectrum
        self.smoothed_spectrum = self.previous_spectrum * self.smoothing + spectrum * (1 - self.smoothing)
        
        # Apply sensitivity to the spectrum
        adjusted_spectrum = self.smoothed_spectrum * self.sensitivity
        
        # Calculate energy (for beat detection)
        self.energy = np.mean(adjusted_spectrum[:self.CHUNK//4]) * 2
        
        return adjusted_spectrum
    
    def setup_colors(self, stdscr):
        # Initialize color pairs for curses
        curses.start_color()
        curses.use_default_colors()
        
        # Check how many colors the terminal supports
        if curses.COLORS < 256:
            # Limited color mode - just set up basic color pairs
            color_count = min(curses.COLORS - 1, 7)  # Reserve 0 for default
            for i in range(color_count):
                curses.init_pair(i + 1, i + 1, -1)  # -1 means default background
        else:
            # Full color mode - create color cube
            color_count = 216  # 6x6x6 color cube
            
            # Create color pairs
            for i in range(color_count):
                # Convert index to r,g,b (0-5 range for each)
                r = (i // 36) % 6
                g = (i // 6) % 6
                b = i % 6
                
                # Scale to 0-1000 range for curses
                r_curses = int((r * 1000) / 5)
                g_curses = int((g * 1000) / 5)
                b_curses = int((b * 1000) / 5)
                
                # Define color and color pair
                curses.init_color(i + 16, r_curses, g_curses, b_curses)
                curses.init_pair(i + 1, i + 16, -1)  # -1 means default background
    
    def run(self, stdscr):
        # Setup curses
        curses.curs_set(0)  # Hide cursor
        
        # Check terminal color support
        has_color = curses.has_colors()
        can_change = curses.can_change_color() if has_color else False
        
        if has_color:
            self.setup_colors(stdscr)
        
        stdscr.timeout(0)  # Non-blocking input
        stdscr.erase()
        
        # Initialize all visualizers
        for visualizer in self.visualizers:
            visualizer.setup()
        
        try:
            while True:
                # Handle keypresses
                try:
                    key = stdscr.getkey()
                    if key == 'q':
                        break
                    elif key == 'm':
                        self.current_visualizer_index = (self.current_visualizer_index + 1) % len(self.visualizers)
                    elif key == ' ':
                        self.pause = not self.pause
                    elif key == '+' or key == '=':  # Both + and = (unshifted +) keys
                        self.sensitivity += self.sensitivity_step
                    elif key == '-':
                        self.sensitivity = max(0.1, self.sensitivity - self.sensitivity_step)
                    else:
                        # Pass the key to the current visualizer
                        current_vis = self.visualizers[self.current_visualizer_index]
                        current_vis.handle_keypress(key)
                except:
                    pass
                
                if not self.pause:
                    # Get current terminal dimensions
                    height, width = stdscr.getmaxyx()
                    
                    # Get audio data
                    spectrum = self.get_audio_data()
                    
                    # Clear screen
                    stdscr.erase()
                    
                    # Update hue offset
                    self.hue_offset = (self.hue_offset + 0.005) % 1.0
                    
                    # Get current visualizer
                    current_vis = self.visualizers[self.current_visualizer_index]
                    
                    # Draw info
                    stdscr.addstr(0, 0, f"Terminal Audio Visualizer | {current_vis.name} | {self.current_visualizer_index+1}/{len(self.visualizers)} | Sensitivity: {self.sensitivity:.1f} | [Q]uit | [M]ode | [+/-] Sensitivity | [Space] Pause")
                    
                    # Draw the current visualization
                    current_vis.draw(stdscr, spectrum, height, width, self.energy, self.hue_offset)
                    
                    # Update screen
                    stdscr.refresh()
                    
                    # Control frame rate
                    time.sleep(0.016)  # ~60fps
        finally:
            # Cleanup
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

# Run the visualizer
if __name__ == "__main__":
    visualizer = TerminalAudioVisualizer()
    curses.wrapper(visualizer.run)