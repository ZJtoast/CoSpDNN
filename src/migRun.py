import subprocess
from multiprocessing import Process
import itertools

def run_script(script_name, neurons, batch, layer , log):
    cmd = [script_name, str(neurons), str(batch), str(layer), str(log)]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    #config
    batch = 60000
    neurons_list = [1024, 4096, 16384, 65536]
    layer_list = [120, 480, 1920]
    
    #homogeneous test
    print("🚀  homogeneous test Start:")
    for neurons, layer in itertools.product(neurons_list, layer_list):
        p1 = Process(target=run_script, 
                     args=('./mig1.sh', neurons, batch, layer,"homo1"))
        p2 = Process(target=run_script, 
                     args=('./mig2.sh', neurons, batch, layer,"homo2"))
        
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        print(f"✅ complete: neurons={neurons}, layer={layer}")

    #heterogeneous test
    print("\n🚀 heterogeneous test Start:")
    for layer in layer_list:

        for (n1, n2) in itertools.permutations(neurons_list, 2):
            p1 = Process(target=run_script, 
                         args=('./mig1.sh', n1, batch, layer,"hete1"))
            p2 = Process(target=run_script, 
                         args=('./mig2.sh', n2, batch, layer,"hete2"))
            
            p1.start()
            p2.start()
            p1.join()
            p2.join()
            print(f"✅ complete: mig1_neurons={n1}, mig2_neurons={n2}, layer={layer}")