from detection.Form import *
import os
import cv2

bad_inputs = []
dir = 'C:/Users/15091/Desktop/InputJPG'

export_dir_dic = {'export_result_img_dir':'C:/Users/15091/Desktop/Output',
                  'export_text_json_dir_from_OCR':'C:/Users/15091/Desktop/text_json',
                  'import_text_json_dir_from_pdf':'C:/Users/15091/Desktop/text_json_from_pdf',
                  'import_rects_json_dir_from_pdf':'C:/Users/15091/Desktop/ele_json_from_pdf',
                  'export_input_dir':'C:/Users/15091/Desktop/input_detection',
                  'export_result_json_dir':'C:/Users/15091/Desktop/result_json'}
file = '7_1.jpg'
img_path = dir + '/' + file
form_compo_detection(img_path, resize_height=None, export_dir=export_dir_dic, text_recover = True, rec_recover=False)

'''for file in os.listdir(dir):
    img_path = dir + '/' + file
    print(img_path)
    try:
        form_compo_detection(img_path, resize_height=None, export_dir=export_dir_dic, text_recover = True, rec_recover=False)
        print()
    except Exception as e:
        tuple=(img_path,str(Exception),repr(e))
        bad_inputs.append(tuple)
        continue

for i in range(len(bad_inputs)):
    print('No.',i+1,'Error')
    print('Image path: ',bad_inputs[i][0])
    print('Error class: ',bad_inputs[i][1])
    print('Error detail: ',bad_inputs[i][2])'''

# form_compo_detection('data/input/8.jpg')
