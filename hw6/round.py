import numpy as np

def format_reconstructed_algos(input_path='reconstructed_algos.csv',
                                output_path='formatted_reconstructed_algos.csv',
                                precision=6):

    data = np.loadtxt(input_path, delimiter=',')


    np.savetxt(output_path, data, delimiter=',', fmt=f'%.{precision}f')

    print(f"Файл сохранён в {output_path} с точностью до {precision} знаков.")

format_reconstructed_algos()
