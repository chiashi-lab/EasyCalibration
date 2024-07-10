import pandas as pd
import os

#backgroundあったら引く
#symphonyの生データのx軸を反転させてラフに合わせる
#ラフに合わせるときのオフセットは経験則なので適宜変更すること
#grating = 150/mm　とする
center_wl = 1200
pixels = 512 #symphonyの横のピクセル数，ちなみに縦は1ピクセルです

def main():
    print('iHR上の中心波長：', end='')
    center_wl = int(input())

    print('データが保存されているフォルダのパスを入力してください：')
    path = str(input())
    pass_list = []
    pass_list = os.listdir(path)
    sorted(pass_list)
    pass_list = [path + '/' + i for i in pass_list]
    numberofdata = len(pass_list)

    print('backgroundはありますか？　y or n：', end='')
    bg_judge = input()
    if bg_judge == 'y':
        print('backgroundファイルのパス：', end='')
        pass_bg = input().strip('"')
        df_bg = pd.read_csv(pass_bg, sep='\t', header=None)

    for i in range(numberofdata):
        df = pd.read_csv(pass_list[i], sep='\t', header=None)

        # backgroundを引く処理
        if bg_judge == 'y':
            df[1] = df[1] - df_bg[1]

        #x軸を反転させる処理
        df_reverse = df.iloc[::-1]

        #x軸にラフな値を代入する処理
        list_wl = []
        # start_wl = center_wl - 319 #319は過去の結果からの概算結果
        start_wl = center_wl - 246  #246は過去の結果からの概算結果
        delta_wl = 509.5 / 511
        for j in range(pixels):
            list_wl.append(start_wl + delta_wl * j)
        df_reverse[0] = list_wl

        #保存
        df_reverse.to_csv(pass_list[i][:-4] + '_reversed.txt', index=False, header=False)
        print(i + 1, '番目のデータが保存されました')

if __name__ =='__main__':
    main()