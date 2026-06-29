import subprocess
import time
import threading
import sys
import os
import matplotlib.pyplot as plt

class GPUEnergyMemoryLogger:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False

        # Energy logging
        self.energy_joules = 0.0
        self.latest_power = 0.0
        self.power_log = []
        self.time_log = []

        # Memory logging
        self.latest_mem = 0.0
        self.peak_mem = 0.0
        self.mem_log = []

        # Time logging
        self.start_time = None
        self.end_time = None

    def _log_gpu_stats(self):
        while self.running:
            timestamp = time.time() - self.start_time
            self.time_log.append(timestamp)

            try:
                # -------- Power Logging --------
                power = subprocess.check_output(
                    "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits",
                    shell=True
                )
                power_watts = float(power.strip())
                self.latest_power = power_watts
                self.energy_joules += power_watts * self.interval
                self.power_log.append(power_watts)

                # -------- Memory Logging --------
                mem = subprocess.check_output(
                    "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                    shell=True
                )
                mem_mb = float(mem.strip())
                self.latest_mem = mem_mb
                self.mem_log.append(mem_mb)
                self.peak_mem = max(self.peak_mem, mem_mb)

                print(
                    f"Power: {power_watts:6.2f} W | "
                    f"Mem: {mem_mb:7.1f} MB | "
                    f"Peak Mem: {self.peak_mem:7.1f} MB",
                    end='\r'
                )

            except Exception as e:
                print(f"Error reading GPU stats: {e}")

            time.sleep(self.interval)

    def _generate_graphs(self):
        # ---- POWER CURVE ----
        plt.figure(figsize=(8,4))
        plt.plot(self.time_log, self.power_log, linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Power (W)")
        plt.title("GPU Power Usage Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("gpu_power_curve.png", dpi=600)
        plt.close()

        # ---- MEMORY CURVE ----
        plt.figure(figsize=(8,4))
        plt.plot(self.time_log, self.mem_log, linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Memory Usage (MB)")
        plt.title("GPU Memory Usage Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("gpu_memory_curve.png", dpi=600)
        plt.close()

    def start(self):
        self.start_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._log_gpu_stats)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.end_time = time.time()

        total_time = self.end_time - self.start_time
        avg_power = self.energy_joules / total_time if total_time > 0 else 0

        # Create power + memory graphs
        self._generate_graphs()

        return self.energy_joules, avg_power, self.peak_mem, total_time


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gpu_logger.py <command_to_run>")
        sys.exit(1)

    logger = GPUEnergyMemoryLogger(interval=1.0)
    logger.start()

    os.system(" ".join(sys.argv[1:]))

    energy, avg_power, peak_mem, runtime = logger.stop()

    print("\n\n--- GPU Usage Summary ---")
    print(f"Total Runtime:            {runtime:.1f} sec")
    print(f"Total Energy Consumed:    {energy:.2f} J")
    print(f"Total Energy Consumed:    {energy/3.6e6:.6f} kWh")
    print(f"Average Power:            {avg_power:.2f} W")
    print(f"Peak Memory Usage:        {peak_mem:.1f} MB")
    print("Graphs Saved:")
    print("  - gpu_power_curve_houstan.png")
    print("  - gpu_memory_curve_houstan.png")
