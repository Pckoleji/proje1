import cv2
#img = cv2.imread('resim1.jpg')
cep= cv2.VideoCapture(0)

cep.set(3,640)#genişlik
cep.set(4,480)#genişlik
listDizin = []
list1 = 'coco.names'
with open(list1,'rt') as f:
    listDizin = f.read().rstrip('\n').split('\n')
dosya1 = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
dosya2 =  'frozen_inference_graph.pb'
atn= cv2.dnn_DetectionModel(dosya1,dosya2)
atn.setInputSize(326,326)
atn.setInputScale(1.0/ 127.5)
atn.setInputMean((127.5, 127.5, 127,5))
atn.setInputSwapRB(True)
while True:
    success,img = cep.read()
    idd, itexf, inxx = atn.detect(img, confThreshold=0.5)
    print(idd,inxx)

    if len(idd) != 0:
        for idd, itef, inx in zip(idd.flatten(), itexf.falletn(), inxx):
            cv2.rectangle(img, inx, color=(0, 255, 0), thickness=5)
            cv2.putText(img, listDizin[idd - 1].upper(), (inx[0] + 10, inx[1] + 30),  # upper büyük harf
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)  # FONT_HERSHEY_TRIPLEX yazı sitili
            cv2.putText(img, str(round(itef*7, 3)), (inx[1] + 50, inx[1] + 70),  # upper büyük harf
                        cv2.FONT_ITALIC, 1, (0, 255, 255), 2)

    cv2.imshow("ekran",img)
    if cv2.waitKey(30) & 0xFF== ord('w'):
       break





