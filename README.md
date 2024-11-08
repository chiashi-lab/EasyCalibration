あなたのキャリブレーションライフを快適に

# 使い方
## 0. 事前準備
pythonとgitをインストールしてください．

https://www.python.org/

https://gitforwindows.org/

pythonのバージョンは3.10.6以上，3.12.0未満を強く推奨します．3.13では動作しません．
pythonインストール時に環境変数パスのチェックを忘れないでください

Gitのインストールは何も考えずにすべて「Next」をクリックしてもらえればOKです．
## 1. ダウンロード
ダウンロードし，適当な場所に展開してください．  
## 2. インストール
install.batをダブルクリックしてください．
## 3. 実行
run.batをダブルクリックしてください．
デスクトップなどの好きなところにショートカットを作成しておくと便利です．
## 4. ラマンやレイリーにおける自動フィッティングでの使い方
### 4-1. 参照スペクトルをインプットする
WiREのソフトでアウトプットしたテキストファイルまたはSolisからアウトプットしたテキストファイルを，`Reference`の領域にドラッグアンドドロップします．  
次に，何の物質のスペクトルなのかを選択し，キャリブレーションの次元を選択します．  
Linear, Quadratic, Cubicの3種類あり，最適な次元がどれなのかは`Help`の領域に記載しています．(迷ったらLinearでOKです．)
### 4-2. キャリブレーションするデータをインプットする
同様に，WiREからアウトプットしたテキストファイルまたはSolisからアウトプットしたテキストファイルをを`Data to calibrate`にドラッグアンドドロップします．  
複数のデータセットを同時にインプット可能です．  
好きなだけデータをドラッグアンドドロップしたら，`CALIBRATE`ボタンでキャリブレーションを実行します．
### 4-3. データのダウンロード
`DOWNLOAD`ボタンからキャリブレーション済のデータをダウンロードできます．  
生データの同じフォルダに`<元のファイル名>_<タイムスタンプ>.txt`という名前で保存されるはずです．
## 5. PLにおける手動フィッティングでの使い方
### 5-1. symphonyで取得したデータからバックグラウンドの削除と反転を行う
同じディレクトリにある`PL_background_subtraction.py`というCLIアプリで行います．`python PL_background_subtraction.py`で起動します。
データが保存されているフォルダのパスや、ArHgの参照スペクトルファイルのパスを入力すると、バックグラウンドの削除と反転を行います。
反転されたデータは、元のデータと同じフォルダに`<元のファイル名>_reversed.txt`という名前で保存されます。
### 5-2. キャリブレーションするデータをインプットする
`python main.py`でEasyCalibrationを起動します．
反転されたデータのテキストファイルを，`Data to calibrate`にドラッグアンドドロップします．  
複数のデータセットを同時にインプット可能です．
### 5-3. 参照スペクトルをインプットする
symphonyで取得したArHgスペクトルのテキストファイルを，`Reference`の領域にドラッグアンドドロップします．  
### 5-4. キャリブレーションする
次に，何の物質のスペクトルなのかを選択し，キャリブレーションの次元を選択します．  
Linear, Quadratic, Cubicの3種類あり，最適な次元がどれなのかは`Help`の領域に記載しています．(迷ったらLinearでOKです．)
`Assign`ボタンを押し、ウインドウを開いたまま赤枠で囲いArHgのスペクトルを割り当てます。囲う幅はなるべく狭くして複数のピークを含めないようにしてください。
スペクトルをいくつか割り当てられたら`CALIBRATE`ボタンでキャリブレーションを実行します．
### 5-5. データのダウンロード
`DOWNLOAD`ボタンからキャリブレーション済のデータをダウンロードできます．  
生データの同じフォルダに`<元のファイル名>_<タイムスタンプ>.txt`という名前で保存されるはずです．
## 6. 更新情報
2022年12月7日：Renishaw Ramanによって取得したデータをサポートしています．\
2023年9月13日追記：488nm Raman (B11A1)はsulfur, naphthalene, BMBをサポートしています．\
2024年7月10日追記：PLをサポートしています. ArHgを参照スペクトルとして使ってください．