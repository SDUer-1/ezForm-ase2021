from detection.Text import Text
from detection.Element import Element
from detection.Image import Image
from detection.Input import Input
from detection.Table import Table
from detection.Row import Row
from detection.Configuration import Configuration
import detection.ocr as ocr

import cv2
import time
import numpy as np
import string
import os
import json
import itertools


def form_compo_detection(form_img_file_name, export_dir, resize_height=None, text_recover=False, rec_recover=False):
    # *** 1. Form structure recognition ***
    form = Form(form_img_file_name, resize_height=resize_height)
    form.element_detection(recover=rec_recover, path=export_dir['import_rects_json_dir_from_pdf'])
    form.text_detection(recover_from_json=text_recover, json_dir=export_dir['export_text_json_dir_from_OCR'])
    #form.visualize_all_elements()
    form.element_refine()
    form.Google_OCR_sentences_recognition()
    form.assign_element_ids()
    # form.img.visualize_elements_contours_individual()
    #form.visualize_all_elements()
    #form.visualize_all_elements_one_by_one(reset=True, opt='org',rec_on=True,line_on=False,squ_on=False)

    # *** 2. Special element recognition ***
    form.border_and_textbox_recognition()
    # form.visualize_all_elements()
    form.character_box_recognition()
    # form.visualize_all_elements()

    # *** 3. Units labelling ***
    form.label_elements_as_units()
    form.sort_units()
    # form.visualize_units()
    form.border_line_recognition()
    form.bar_units_refine()
    #form.visualize_units()

    # *** 4. Form structure recognition ***
    form.check_vertical_aligned_form()
    # form.visualize_vertical_separators()
    form.group_units_by_separators()
    # form.visualize_unit_groups()

    # *** 5. Table obj ***
    form.table_detection()
    # form.visualize_table()
    form.table_refine()
    # form.visualize_table()

    # *** 6. Input compound recognition ***
    form.input_compound_recognition()
    form.input_refine()
    # form.visualize_inputs_one_by_one()
    form.text_refine()
    # form.visualize_detection_result()
    '''form.visualize_detection_result_one_by_one(reset=True,
                                              opt='org',text_test=True,rec_test=True,
                                              line_test=True,
                                              table_test=True,input_test=True)'''

    # *** 7. Export ***
    form.export_visualize_inputs_and_tables(export_dir=export_dir['export_input_dir'])
    form.export_detection_result_json(export_dir=export_dir['export_result_json_dir'])
    form.export_detection_result_img(export_dir['export_result_img_dir'])
    return form


class Form:
    def __init__(self, img_file_name, resize_height=None):
        self.config = Configuration()
        self.img_file_name = img_file_name
        self.resize_height = resize_height
        self.img = Image(img_file_name, resize_height=resize_height, configuration=self.config)
        self.form_name = img_file_name.split('/')[-1][:-4]

        # atomic elements
        self.texts = []  # detected by ocr
        self.rectangles = []  # detected by cv
        self.squares = []
        self.lines = []  # detected by cv
        # compound elements
        self.tables = []  # recognize by grouping rectangles
        self.inputs = []  # input elements that consists of guide text (text|textbox) and input filed (rectangle|line)
        self.row_id = 0
        self.table_id = 0

        # units for input, grouped from the above elements
        self.text_units = []  # text (not in box) + textbox
        self.bar_units = []  # rectangles (not textbox) + lines + tables
        self.all_units = []
        self.sorted_left_unit = []
        self.sorted_right_unit = []
        self.sorted_top_unit = []
        self.sorted_bottom_unit = []

        self.vertical_separators = None  # dictionary {left, right, top, bottom}, set None if form is vertical alignment
        self.unit_groups = []  # 3-d list, groups of units segmented by separators, [[[sep1-left-group], [sep1-right-group], [sep1-top-group], [sep1-bottom-group]]]

        self.detection_result_img = None
        self.export_dir = 'data/output/' + self.form_name
        os.makedirs(self.export_dir, exist_ok=True)

    '''
    ****************************
    *** Check Form Structure ***
    ****************************
    '''

    def check_vertical_aligned_form(self):
        '''
        Check if the form is vertical aligned
        :return: set self.vertical_separator if the form is in vertical alignment
        '''

        def check_gaps_from_mid(binary):
            '''
            Check continuously connected gap in columns from the middle leftwards and rightwards
            :param binary: binary map of the form image
            :return: {'left': {col_id1: [(gap1_top, gap1_bottom), (gap2_top, gap2_bottom)],
                    'right': {col_id1: [(gap1_top, gap1_bottom), (gap2_top, gap2_bottom)],
                    'mid': {mid_col_id: [(gap1_top, gap1_bottom), (gap2_top, gap2_bottom)]}}
            '''

            def check_gaps_in_a_col(col):
                col_gaps = []
                gap_top = -1
                gap_bottom = -1
                for i in range(height - 1):
                    if binary[i, col] == 0:
                        if gap_top == -1:
                            gap_top = i
                    else:
                        if gap_top != -1:
                            gap_bottom = i - 1
                            if gap_bottom - gap_top > height / 3:
                                col_gaps.append((gap_top, gap_bottom))
                            gap_top = -1
                            gap_bottom = -1
                if gap_bottom <= gap_top:
                    gap_top = max(0, gap_top)
                    gap_bottom = height - 1
                    if gap_bottom - gap_top > height / 3:
                        col_gaps.append((gap_top, gap_bottom))
                return col_gaps

            (height, width) = binary.shape
            mid = int(width / 2)
            right = mid + 1
            left = mid - 1
            gap_mid = check_gaps_in_a_col(mid)
            gap_right = check_gaps_in_a_col(right)
            gap_left = check_gaps_in_a_col(left)
            gaps = {'mid': {mid: gap_mid}, 'left': {}, 'right': {}}

            spreading = True
            while spreading:
                spreading = False
                if len(gap_right) > 0:
                    gaps['right'][right] = gap_right
                    right = right + 1
                    if right < width - 1:
                        gap_right = check_gaps_in_a_col(right)
                        spreading = True
                if len(gap_left) > 0:
                    gaps['left'][left] = gap_left
                    left = left - 1
                    if left > 0:
                        gap_left = check_gaps_in_a_col(left)
                        spreading = True
            return gaps

        def merge_gaps_as_separators(gaps):
            '''
            merge the detected gaps as vertical separators
            :return: list of separators: [{'top', 'bottom', 'left', 'right'}]
            '''
            gaps_m = gaps['mid']
            gaps_left = gaps['left']
            gaps_right = gaps['right']
            mid_col_id = list(gaps_m.keys())[0]
            left_col_ids = sorted(list(gaps_left.keys()), reverse=True)
            right_col_ids = sorted(list(gaps_right.keys()))

            gm = gaps_m[mid_col_id]
            merged_gap = {}
            for g in gm:
                merged_gap[g] = {'left': mid_col_id, 'right': mid_col_id, 'top': g[0], 'bottom': g[1]}

            for i in left_col_ids:
                gl = gaps_left[i]
                # match all gaps between gaps of the mid col and gaps in this col
                for a in gm:
                    for b in gl:
                        if abs(a[0] - b[0]) < 10 and abs(a[1] - b[1]) < 10:
                            if merged_gap[a]['left'] - i == 1:
                                merged_gap[a]['left'] = i
                                merged_gap[a]['top'] = max(merged_gap[a]['top'], b[0])
                                merged_gap[a]['bottom'] = min(merged_gap[a]['bottom'], b[1])

            for i in right_col_ids:
                gl = gaps_right[i]
                # match all gaps between gaps of the mid col and gaps in this col
                for a in gm:
                    for b in gl:
                        if abs(a[0] - b[0]) < 10 and abs(a[1] - b[1]) < 10:
                            if i - merged_gap[a]['right'] == 1:
                                merged_gap[a]['right'] = i
                                merged_gap[a]['top'] = max(merged_gap[a]['top'], b[0])
                                merged_gap[a]['bottom'] = min(merged_gap[a]['bottom'], b[1])

            # reformat as list of separators: [{'top', 'bottom', 'left', 'right'}]
            separators = []
            for k in merged_gap:
                separators.append(merged_gap[k])
            return separators

        all_gaps = check_gaps_from_mid(self.img.binary_map)
        # print(all_gaps)
        separators = merge_gaps_as_separators(all_gaps)
        if len(separators) > 0:
            print('*** The form is vertical alignment with vertical separators:', separators, '***')
            self.vertical_separators = separators
        else:
            print('*** The form is not vertical alignment ***')
            self.vertical_separators = None

    def group_units_by_separators(self):
        '''
        If the form is vertical alignment Group all units by separators
        For each separator, it can segment four groups of units [[left-group], [right-group], [top-group], [bottom-group]]
        :return: [[[sep1-left-group], [sep1-right-group], [sep1-top-group], [sep1-bottom-group]]]
        '''
        # only for vertical aligned form
        if self.vertical_separators is None:
            return
        seps = self.vertical_separators
        groups = []
        for i in range(len(seps)):
            groups.append([[], [], [], []])

        for p, unit in enumerate(self.all_units):
            for i, sep in enumerate(seps):
                if unit.location['bottom'] <= sep['top']:
                    if i == 0 or unit.location['top'] > seps[i - 1]['bottom']:
                        unit.unit_group_id = i * 4 + 0
                        groups[i][0].append(unit)
                elif sep['top'] < unit.location['top'] and unit.location['bottom'] <= sep['bottom']:
                    if unit.location['right'] <= sep['left']:
                        unit.unit_group_id = i * 4 + 1
                        groups[i][1].append(unit)
                    elif unit.location['left'] > sep['right']:
                        unit.unit_group_id = i * 4 + 2
                        groups[i][2].append(unit)
                else:
                    unit.unit_group_id = i * 4 + 3
                    groups[i][3].append(unit)
        self.unit_groups = groups

    '''
    **************************
    *** Element Processing ***
    **************************
    '''

    def get_all_elements(self):
        return self.texts + self.rectangles + self.squares + self.lines + self.tables

    def assign_element_ids(self):
        '''
        Assign an unique id to each element and store the id mapping
        '''
        for i, ele in enumerate(self.get_all_elements()):
            ele.id = i

    def get_detection_result(self):
        '''
        Get all non-noisy independent elements (not in any modules) and modules (table, input compound)
        :return: A list of Elements
        '''
        detection_result = []
        for text in self.texts:
            if not text.in_box and not text.is_abandoned and not text.is_module_part:
                detection_result.append(text)

        for rec in self.rectangles:
            if not rec.is_abandoned and not rec.is_module_part:
                detection_result.append(rec)

        for squ in self.squares:
            if not squ.is_abandoned and not squ.is_module_part:
                detection_result.append(squ)

        for line in self.lines:
            if not line.is_abandoned and not line.is_module_part:
                detection_result.append(line)

        for table in self.tables:
            detection_result.append(table)

        for ipt in self.inputs:
            detection_result.append(ipt)

        return detection_result

    def sort_units(self):
        '''
        Sort all units by left and top respectively, and store the result in id lists
        '''
        self.sorted_left_unit = sorted(self.all_units, key=lambda x: x.location['left'])
        self.sorted_right_unit = self.sorted_left_unit.copy()
        self.sorted_right_unit.reverse()

        self.sorted_top_unit = sorted(self.all_units, key=lambda x: x.location['top'])
        self.sorted_bottom_unit = self.sorted_top_unit.copy()
        self.sorted_bottom_unit.reverse()

    def label_elements_as_units(self):
        '''
        text_units: text (not contained) + textbox
        bar_units: rectangles (not textbox) + lines + tables
        '''
        self.text_units = []  # text (not in box) + textbox
        self.bar_units = []  # rectangles (not textbox) + lines + tables
        self.all_units = []
        for text in self.texts:
            if not text.in_box and not text.is_abandoned:
                text.unit_type = 'text_unit'
                self.text_units.append(text)
        for ele in self.rectangles + self.squares + self.lines:
            if ele.is_abandoned:
                continue
            if ele.type in ('line', 'rectangle', 'square'):
                ele.unit_type = 'bar_unit'
                self.bar_units.append(ele)
            elif ele.type == 'textbox':
                ele.unit_type = 'text_unit'
                self.text_units.append(ele)
        self.all_units = self.text_units + self.bar_units

    def bar_units_refine(self):
        for ele in self.rectangles + self.squares + self.lines:
            if ele.type == 'textbox':
                not_textbox = False
                contain_layer = 1
                ele.contains = sorted(ele.contains, key=lambda x: x.location['top'])
                max_width_layer = ele.contains[0].width
                current_width = ele.contains[0].width
                if len(ele.contains) > 1:
                    for i in range(1, len(ele.contains)):
                        if abs(ele.contains[i].location['top'] - ele.contains[i - 1].location['top']) > 0.4 * max(
                                ele.contains[i].height, ele.contains[i - 1].height) and \
                                abs(ele.contains[i].location['bottom'] - ele.contains[i - 1].location[
                                    'bottom']) > 0.4 * max(
                            ele.contains[i].height, ele.contains[i - 1].height):
                            if current_width > max_width_layer:
                                max_width_layer = current_width
                            current_width = ele.contains[i].width
                        current_width = current_width + ele.contains[i].width
                if abs(ele.location['top'] - min([c.location['top'] for c in ele.contains])) >= 2 * max(
                        [c.height for c in ele.contains]) or \
                        abs(ele.location['bottom'] - max([c.location['bottom'] for c in ele.contains])) >= 2 * max(
                    [c.height for c in ele.contains]):
                    not_textbox = True
                if max_width_layer / ele.width < 0.5:
                    not_textbox = True
                if not_textbox == True:
                    '''neighbour_top = self.find_neighbour_unit(ele, 'top',
                                                             connect_bias=self.config.input_compound_recognition_neighbor2_connect_bias,
                                                             align_bias=self.config.input_compound_recognition_neighbor2_align_bias)
                    neighbour_left = self.find_neighbour_unit(ele, 'left',
                                                             connect_bias=self.config.input_compound_recognition_neighbor2_connect_bias,
                                                             align_bias=self.config.input_compound_recognition_neighbor2_align_bias)
                        if (neighbour_top is not None and neighbour_top.unit_type == 'text_unit' and neighbour_top.in_table is None and\
                            ele.location['top'] - neighbour_top.location['bottom'] < self.config.input_compound_recognition_max_gap_v) or\
                        (neighbour_left is not None and neighbour_left.unit_type == 'text_unit' and neighbour_left.in_table is None and\
                            ele.location['left'] - neighbour_left.location['right'] < self.config.input_compound_recognition_max_gap_v):
                        ele.unit_type = 'bar_unit'''
                    ele.unit_type = 'bar_unit'
                    self.text_units.remove(ele)
                    self.bar_units.append(ele)

    def find_neighbour_unit(self, unit, direction='right', connect_bias=10, align_bias=4):
        '''
        Find the first unit 1.next to and 2.in alignment with the target
        :param direction:
            -> left: find left neighbour
            -> right: find right neighbour
            -> top: find top neighbour
            -> below: find below neighbour
        :return:
        '''
        if direction == 'right':
            if unit.neighbour_right is not None:
                return unit.neighbour_right
            # check is there any connected unit on the right
            for u in self.sorted_left_unit:
                if u.id != unit.id and u.unit_group_id == unit.unit_group_id and \
                        u.location['left'] + connect_bias >= unit.location['right']:
                    # the tow should be justified
                    if unit.is_in_alignment(u, direction='h', bias=align_bias):
                        unit.neighbour_right = u
                        u.neighbour_left = unit
                        return u
        elif direction == 'left':
            if unit.neighbour_left is not None:
                return unit.neighbour_left
            # check is there any connected unit on the left
            for u in self.sorted_right_unit:
                if u.id != unit.id and u.unit_group_id == unit.unit_group_id and \
                        unit.location['left'] + connect_bias >= u.location['right']:
                    # the tow should be justified
                    if unit.is_in_alignment(u, direction='h', bias=align_bias):
                        unit.neighbour_left = u
                        u.neighbour_right = unit
                        return u
        elif direction == 'below':
            if unit.neighbour_bottom is not None:
                return unit.neighbour_bottom
            # check is there any connected unit below
            for u in self.sorted_top_unit:
                if u.id != unit.id and u.unit_group_id == unit.unit_group_id and \
                        u.location['top'] + connect_bias >= unit.location['bottom']:
                    # the tow should be justified if they are neighbours
                    if unit.is_in_alignment(u, direction='v', bias=align_bias):
                        unit.neighbour_bottom = u
                        u.neighbour_top = unit
                        return u
        elif direction == 'top':
            if unit.neighbour_top is not None:
                return unit.neighbour_top
            # check is there any connected unit above
            for u in self.sorted_bottom_unit:
                if u.id != unit.id and u.unit_group_id == unit.unit_group_id and \
                        unit.location['top'] + connect_bias >= u.location['bottom']:
                    # the tow should be justified if they are neighbours
                    if unit.is_in_alignment(u, direction='v', bias=align_bias):
                        unit.neighbour_top = u
                        u.neighbour_bottom = unit
                        return u
        return None

    '''
    *************************
    *** Element Detection ***
    *************************
    '''

    def text_detection(self, json_dir, method='Google', recover_from_json=False):
        if method == 'Baidu':
            self.Baidu_OCR_text_detection()
        elif method == 'Google' and recover_from_json == False:
            self.Google_OCR_text_detection(json_dir=json_dir)
        elif method == 'Google' and recover_from_json == True:
            self.recover_from_json(json_dir=json_dir)
            print('Read from json')
        self.shrink_text_and_filter_noises()

    def recover_from_json(self, json_dir):
        with open(os.path.join(json_dir, self.form_name + '.json')) as f:
            texts_json = json.load(f)
        for text in texts_json:
            self.texts.append(Text(text['content'], text['location']))

    def Baidu_OCR_text_detection(self):
        start = time.clock()
        detection_result = ocr.ocr_detection_baidu(self.img_file_name)
        texts = detection_result['words_result']
        for text in texts:
            location = {'left': text['location']['left'], 'top': text['location']['top'],
                        'right': text['location']['left'] + text['location']['width'],
                        'bottom': text['location']['top'] + text['location']['height']}
            self.texts.append(Text(text['words'], location))
        print('*** Baidu OCR Processing Time:%.3f s***' % (time.clock() - start))

    def Google_OCR_text_detection(self, json_dir):
        start = time.clock()
        detection_results = ocr.ocr_detection_google(self.img.img)
        if detection_results is not None:
            for result in detection_results:
                x_coordinates = []
                y_coordinates = []
                text_location = result['boundingPoly']['vertices']
                text = result['description']
                for loc in text_location:
                    x_coordinates.append(loc['x'])
                    y_coordinates.append(loc['y'])
                location = {'left': min(x_coordinates), 'top': min(y_coordinates),
                            'right': max(x_coordinates), 'bottom': max(y_coordinates)}
                self.texts.append(Text(text, location))
        text_json_list = []
        for text in self.texts:
            text_json = {'type': text.type, 'content': text.content, 'location': text.location}
            text_json_list.append(text_json)
        json.dump(text_json_list, open(os.path.join(json_dir, self.form_name + '.json'), 'w+'), indent=4)
        print('Text JSON Write to:', os.path.join(json_dir, self.form_name + '.json'))
        print('*** Google OCR Processing Time:%.3f s***' % (time.clock() - start))

    def Google_OCR_sentences_recognition(self):
        '''
        Merge separate words detected by Google ocr into a sentence
        '''
        changed = True
        while changed:
            changed = False
            temp_set = []
            self.texts = sorted(self.texts, key=lambda x: x.location['left'])
            for text_a in self.texts:
                merged = False
                for text_b in temp_set:
                    self.config.set_sentences_recognition_parameter(text_a, text_b)
                    if text_a.is_on_same_line(text_b, 'h', bias_justify=self.config.sentences_recognition_bias_justify,
                                              bias_gap=self.config.sentences_recognition_bias_gap):
                        text_b.merge_text(text_a)
                        merged = True
                        changed = True
                        break
                if not merged:
                    temp_set.append(text_a)
            self.texts = temp_set.copy()

    def shrink_text_and_filter_noises(self):
        noises = []
        for text in self.texts:
            text.shrink_bound(self.img.binary_map)
            if min(text.width, text.height) <= self.config.shrink_text_and_filter_noises:
                text.is_abandoned = True
                noises.append(text)
        for n in noises:
            self.texts.remove(n)

    def element_detection(self, recover=False, path=None):
        start = time.clock()
        if recover == False:
            self.rectangles, self.squares = self.img.detect_rectangle_and_square_elements(configuration=self.config)
            #self.lines = self.img.detect_line_elements(configuration=self.config)
            self.filter_detection_noises()
            print('*** Element Detection Time:%.3f s***' % (time.clock() - start))
        else:
            self.recover_rectangles_from_json(path=path)
            self.filter_detection_noises()

    def recover_rectangles_from_json(self, path):
        with open(os.path.join(path, self.form_name + '.json')) as f:
            ele_json = json.load(f)
        # recover rectangle from curve
        rectangles = self.rectangles.copy()
        for curve in ele_json['curve_list']:
            if 12 >= len(curve['points']) >= 4 and \
                    (curve['location']['right'] - curve['location']['left']) * (
                    curve['location']['bottom'] - curve['location']['top']) > 100:
                curve_ele = Element(location=curve['location'])
                if abs(curve_ele.width - curve_ele.height) < 3:
                    curve_ele.type = 'square'
                    self.squares.append(curve_ele)
                else:
                    curve_ele.type = 'rectangle'
                    self.rectangles.append(curve_ele)

        vertical_list = []
        horizontal_list = []
        for rec in ele_json['rec_list']:
            if rec['location']['right'] - rec['location']['left'] <= 3 and \
                rec['location']['bottom'] - rec['location']['top'] > 3:
                rec['type'] = 'line'
                vertical_list.append(rec)
            elif rec['location']['bottom'] - rec['location']['top'] < 3 and \
                    rec['location']['right'] - rec['location']['left'] > 3:
                rec['type'] = 'line'
                horizontal_list.append(rec)
            elif rec['location']['bottom'] - rec['location']['top'] > 3 and \
                    rec['location']['right'] - rec['location']['left'] > 3:
                if abs((rec['location']['bottom'] - rec['location']['top']) - (rec['location']['right'] - rec['location']['left'])) < 3:
                    self.squares.append(Element(type='square', location=rec['location']))
                else:
                    self.rectangles.append(Element(type='rectangle', location=rec['location']))
        for line in ele_json['line_list']:
            if line['height'] > line['width']:
                vertical_list.append(line)
            else:
                horizontal_list.append(line)
        vertical_list = sorted(vertical_list, key=lambda x: x['location']['left'])
        horizontal_list = sorted(horizontal_list, key=lambda x: x['location']['top'])
        maybe_vertical_list = []
        maybe_horizontal_list = []
        for i in itertools.combinations(vertical_list, 2):
            if i[0] != i[1] and abs(i[0]['location']['top'] - i[1]['location']['top']) < 3 and abs(
                    i[0]['location']['bottom'] - i[1]['location']['bottom']) < 3:
                maybe_vertical_list.append(i)
        for i in itertools.combinations(horizontal_list, 2):
            if i[0] != i[1] and abs(i[0]['location']['left'] - i[1]['location']['left']) < 3 and abs(
                    i[0]['location']['right'] - i[1]['location']['right']) < 3:
                maybe_horizontal_list.append(i)
        for v_pair in maybe_vertical_list:
            for h_pair in maybe_horizontal_list:
                if abs(v_pair[0]['location']['top'] - h_pair[0]['location']['bottom']) < 3 and abs(
                        v_pair[0]['location']['right'] - h_pair[0]['location']['left']) < 3 and \
                        abs(v_pair[1]['location']['bottom'] - h_pair[1]['location']['top']) < 3 and abs(
                    v_pair[1]['location']['left'] - h_pair[0]['location']['right']) < 3:
                    rec_location = {'left': v_pair[0]['location']['left'],
                                    'top': h_pair[0]['location']['top'],
                                    'right': v_pair[1]['location']['right'],
                                    'bottom': h_pair[1]['location']['bottom']}
                    if abs((rec_location['right'] - rec_location['left']) - (rec_location['bottom'] - rec_location['top'])) < 3:
                        self.squares.append(Element(type='square', location=rec_location))
                    else:
                        self.rectangles.append(Element(type='rectangle', location=rec_location))

        # remove redundant rectangles
        for rec_pair in itertools.combinations(self.rectangles, 2):
            if abs(rec_pair[0].location['left'] - rec_pair[1].location['left']) < 5 and \
                    abs(rec_pair[0].location['right'] - rec_pair[1].location['right']) < 5 and \
                    abs(rec_pair[0].location['top'] - rec_pair[1].location['top']) < 5 and \
                    abs(rec_pair[0].location['bottom'] - rec_pair[1].location['bottom']) < 5:
                if rec_pair[1] in self.rectangles:
                    self.rectangles.remove(rec_pair[1])

        # line seperate rectangles
        for h_line in horizontal_list:
            for rec in self.rectangles:
                if h_line['location']['top'] - rec.location['top'] > 10 and \
                        rec.location['bottom'] - h_line['location']['bottom'] > 10 and \
                        h_line['location']['left'] - rec.location['left'] < 3 and \
                        rec.location['right'] - h_line['location']['right'] < 3:
                    new_rec_1_location = {'left': rec.location['left'],
                                          'top': rec.location['top'],
                                          'right': rec.location['right'],
                                          'bottom': h_line['location']['bottom']}
                    new_rec_2_location = {'left': rec.location['left'],
                                          'top': h_line['location']['bottom'],
                                          'right': rec.location['right'],
                                          'bottom': rec.location['bottom']}
                    new_rec_1 = Element(type='rectangle', location=new_rec_1_location)
                    new_rec_2 = Element(type='rectangle', location=new_rec_2_location)
                    self.rectangles.append(new_rec_1)
                    self.rectangles.append(new_rec_2)
                    self.rectangles.remove(rec)
        for v_line in vertical_list:
            for rec in self.rectangles:
                if v_line['location']['left'] - rec.location['left'] > 10 and \
                        rec.location['right'] - v_line['location']['right'] > 10 and \
                        v_line['location']['top'] - rec.location['top'] < 3 and \
                        rec.location['bottom'] - v_line['location']['bottom'] < 3:
                    new_rec_1_location = {'left': rec.location['left'],
                                          'top': rec.location['top'],
                                          'right': v_line['location']['right'],
                                          'bottom': rec.location['bottom']}
                    new_rec_2_location = {'left': v_line['location']['right'],
                                          'top': rec.location['top'],
                                          'right': rec.location['right'],
                                          'bottom': rec.location['bottom']}
                    new_rec_1 = Element(type='rectangle', location=new_rec_1_location)
                    new_rec_2 = Element(type='rectangle', location=new_rec_2_location)
                    self.rectangles.append(new_rec_1)
                    self.rectangles.append(new_rec_2)
                    self.rectangles.remove(rec)


    def element_refine(self):
        for text in self.texts:
            text_noise = True
            for c in text.content:
                if c != 'O' and c != 'D' and c != 'ロ':
                    text_noise = False
            if text_noise == True:
                self.texts.remove(text)
                continue
            for ele in self.rectangles + self.squares + self.lines:
                if text.pos_relation(element=ele, pos_bias=0) == 1:
                    if text.height - ele.height > 10:
                        if ele.type == 'rectangle':
                            self.rectangles.remove(ele)
                            continue
                        if ele.type == 'square':
                            self.squares.remove(ele)
                            continue
                        if ele.type == 'line':
                            self.lines.remove(ele)
                            continue
                if abs(text.location['top'] - ele.location['top']) < self.config.element_refine_bias and abs(
                        text.location['bottom'] - ele.location['bottom']) < self.config.element_refine_bias and \
                        abs(text.location['left'] - ele.location['left']) < self.config.element_refine_bias and abs(
                    text.location['right'] - ele.location['right']) < self.config.element_refine_bias:
                    if text.content == 'O' or text.content == 'D':
                        self.texts.remove(text)
                        break
                    else:
                        if ele.type == 'rectangle':
                            self.rectangles.remove(ele)
                            continue
                        if ele.type == 'square':
                            self.squares.remove(ele)
                            continue
                        if ele.type == 'line':
                            self.lines.remove(ele)
                            continue
                if abs(text.location['top'] - ele.location['top']) < self.config.element_refine_bias and abs(
                        text.location['bottom'] - ele.location['bottom']) < self.config.element_refine_bias:
                    if len(text.content) > 2:
                        wrong_width = int(text.width / len(text.content))
                    else:
                        wrong_width = 1
                    if abs(min(text.location['left'], ele.location['left']) - ele.location['left']) < abs(
                            max(text.location['right'], ele.location['right']) - ele.location['right']) and \
                            text.pos_relation(element=ele, pos_bias=0) != 0:
                        text.location['left'] = ele.location['right'] + wrong_width
                        text.width = text.location['right'] - text.location['left']
                        text.area = text.width * text.height
                        '''if len(text.content) > 1:
                            text.content = text.content[1:]'''
                        continue
                    if abs(min(text.location['left'], ele.location['left']) - ele.location['left']) >= abs(
                            max(text.location['right'], ele.location['right']) - ele.location[
                                'right']) and text.pos_relation(element=ele, pos_bias=0) != 0:
                        text.location['right'] = ele.location['left'] - wrong_width
                        text.width = text.location['right'] - text.location['left']
                        text.area = text.width * text.height
                        '''if len(text.content) > 1:
                            text.content = text.content[1:]'''
                        continue
                if text.pos_relation(element=ele, pos_bias=0) == 2:
                    if len(text.content) > 1:
                        bias = int(text.width / len(text.content))
                    else:
                        bias = 1
                    if text.location['left'] <= ele.location['left'] <= text.location['right'] and (
                            text.location['top'] > ele.location['top'] and text.location['bottom'] < ele.location[
                        'bottom']):
                        text.location['left'] = ele.location['left'] + bias
                        text.width = text.location['right'] - text.location['left']
                        text.area = text.width * text.height
                        continue
                    if text.location['left'] <= ele.location['right'] <= text.location['right'] and (
                            text.location['top'] > ele.location['top'] and text.location['bottom'] < ele.location[
                        'bottom']):
                        text.location['right'] = ele.location['right'] - bias
                        text.width = text.location['right'] - text.location['left']
                        text.area = text.width * text.height
                        continue

    def filter_detection_noises(self):
        # count shapes contained in text as noise
        rects = self.rectangles.copy()
        squs = self.squares.copy()
        lines = self.lines.copy()
        for text in self.texts:
            for rec in self.rectangles:
                if text.pos_relation(rec, self.config.filter_detection_noises_pos_relation_bias) == 1:
                    rects.remove(rec)
            for line in self.lines:
                if text.pos_relation(line, self.config.filter_detection_noises_pos_relation_bias) == 1:
                    lines.remove(line)
            # if a square is in a text, store it in text.contain_square
            for squ in self.squares:
                if squ.pos_relation(text,
                                    self.config.filter_detection_noises_pos_relation_bias) == -1 and squ.area / text.area < 0.6:
                    squ.nesting_text = text
            self.rectangles = rects.copy()
            self.squares = squs.copy()
            self.lines = lines.copy()

        # filter out double nested shapes
        redundant_nesting = []
        rect_squs = self.rectangles + self.squares
        for i, rect_squ in enumerate(rect_squs):
            containment_area = 0
            containments = []
            for j in range(i + 1, len(rect_squs)):
                ioi, ioj = rect_squ.calc_intersection(rect_squs[j])
                if ioj == 1:
                    containment_area += rect_squs[j].area
                    containments.append(rect_squs[j])
            # if containment_area / rect_squ.area > 0:
            #     print(len(containments), containment_area, rect_squ.area, containment_area / rect_squ.area)
            #     rect_squ.visualize_element(self.get_img_copy(), show=True)
            if containment_area / rect_squ.area > 0.5:
                rect_squ.is_abandoned = True
                redundant_nesting.append(rect_squ)
        for r in redundant_nesting:
            if r.type == 'rectangle':
                self.rectangles.remove(r)
            elif r.type == 'square':
                self.squares.remove(r)

    '''
    ***********************************
    *** Special Element Recognition ***
    ***********************************
    '''

    def border_and_textbox_recognition(self):
        '''
        If a rectangle contains only texts in it, then label the rect as type of 'textbox'
        Else if it contains other rectangles in it, then label it as type of 'border'
        '''
        all_eles = self.get_all_elements()
        # iteratively check the relationship between eles and rectangles
        for ele in all_eles:
            for rec_squ in self.rectangles + self.squares:
                if ele.id == rec_squ.id:
                    continue
                relation = ele.pos_relation(rec_squ, self.config.border_and_textbox_recognition_pos_relation_bias)
                # if the element is contained in the rectangle box
                if relation == -1:
                    # if rec_squ not in ele.contains:
                    rec_squ.contains.append(ele)
                    rec_squ.containment_area += ele.area

        for rec_squ in self.rectangles + self.squares:
            rs_type = rec_squ.is_textbox_or_border(self.config.is_textbox_or_border_ratio)
            # merge text vertically for a textbox
            if rs_type == 'textbox':
                for containment in rec_squ.contains:
                    containment.in_box = True
                rec_squ.textbox_merge_and_extract_texts_content(alignment_bias=self.config.textbox_merge_alignment_bias,
                                                                v_max_merged_gap=self.config.textbox_merge_v_max_merged_gap)

    def border_line_recognition(self):
        '''
        Recognize if a rectangle/line is a nonfunctional border line
        '''
        borders = []
        for bar in self.bar_units:
            if bar.type == 'line':
                neighbour = self.find_neighbour_unit(bar, direction='top',
                                                     connect_bias=self.config.border_line_recognition_neighbor_connect_bias,
                                                     align_bias=self.config.border_line_recognition_neighbor_align_bias)
                if neighbour is not None and (abs(neighbour.location['left'] - bar.location[
                    'left']) > self.config.border_line_recognition_left_bias or \
                        bar.location['top'] - neighbour.location[
                    'bottom'] < self.config.border_line_recognition_top_bias):
                    bar.type = 'border'
                    bar.unit_type = None
                    borders.append(bar)
        for bar in borders:
            self.bar_units.remove(bar)
            self.all_units.remove(bar)
        self.sort_units()

    def character_box_recognition(self):
        '''
        Recognize if some rectangles and squares can combine into a character box
        '''
        rect_squs = self.rectangles + self.squares

        changed = True
        while changed:
            changed = False
            temp_set = []
            for r1 in rect_squs:
                merged = False
                for r2 in temp_set:
                    if r2.is_in_same_character_box(r1, self.config) and \
                            (r1.character_num != 1 or r1.area <= self.img.img_shape[0] * self.img.img_shape[
                                1] / 800) and \
                            (r2.character_num != 1 or r2.area <= self.img.img_shape[0] * self.img.img_shape[1] / 800):
                        r2.character_box_merge_ele(r1)
                        merged = True
                        changed = True
                        break
                if not merged:
                    temp_set.append(r1)
            rect_squs = temp_set.copy()

        self.rectangles = []
        self.squares = []
        for rect_squ in rect_squs:
            if rect_squ.type in ('rectangle', 'textbox'):
                self.rectangles.append(rect_squ)
            elif rect_squ.type == 'square':
                self.squares.append(rect_squ)

    '''
    *************************************
    *** Compound Components Detection ***
    *************************************
    '''

    def input_compound_recognition(self):
        '''
        Recognize input unit that consists of [guide text] and [input field]
        First. recognize guide text for input:
            If a text_unit's closet element in alignment is bar_unit, then count it as a guide text
        Second. compound the guide text and its bar unit (input field) as an Input element
        '''
        # *** 4. Checkbox: a square following/followed by a guide text
        for bar in self.bar_units:
            if bar.type == 'square' and bar.in_input is None and bar.in_table is None and len(bar.contains) == 0:
                if bar.nesting_text is not None:
                    self.inputs.append(Input(bar.nesting_text, bar, is_checkbox=True, input_type=4))
                    continue
                # check square's left and right, and chose the
                neighbour_right = self.find_neighbour_unit(bar, direction='right',
                                                           connect_bias=self.config.input_compound_recognition_neighbor4_connect_bias,
                                                           align_bias=self.config.input_compound_recognition_neighbor4_align_bias)
                neighbour_left = self.find_neighbour_unit(bar, direction='left',
                                                          connect_bias=self.config.input_compound_recognition_neighbor4_connect_bias,
                                                          align_bias=self.config.input_compound_recognition_neighbor4_align_bias)
                if neighbour_right is not None and neighbour_right.type == 'text' and neighbour_right.in_input is None and neighbour_right.in_table is None:
                    if neighbour_left is not None and neighbour_left.type == 'text' and neighbour_left.in_input is None and neighbour_left.in_table is None:
                        # check the closer text as guidetext
                        if neighbour_right.location['left'] - bar.location['right'] > bar.location['left'] - \
                                neighbour_left.location['right']:
                            self.inputs.append(Input(neighbour_left, bar, is_checkbox=True, input_type=4))
                        else:
                            self.inputs.append(Input(neighbour_right, bar, is_checkbox=True, input_type=4))
                    else:
                        self.inputs.append(Input(neighbour_right, bar, is_checkbox=True, input_type=4))
                else:
                    if neighbour_left is not None and neighbour_left.type == 'text' and neighbour_left.in_input is None and neighbour_left.in_table is None:
                        self.inputs.append(Input(neighbour_left, bar, is_checkbox=True, input_type=4))

        # *** 1. Embedded Input: input field and guiding text in the same rectangle ***
        for textbox in self.all_units:
            if textbox.type == 'textbox' and textbox.in_input is None and textbox.in_table is None:
                if 0 < len(textbox.contains) <= 2:
                    guiding_text = textbox.contains[0]
                    content = guiding_text.content
                    if content.count(':') == 1 and content.count('.') <= 1:
                        '''neighbour_right = self.find_neighbour_unit(textbox, direction='right', connect_bias=10,align_bias=4)
                        neighbour_bottom = self.find_neighbour_unit(textbox, direction='bottom', connect_bias=10,align_bias=4)
                        embedded = False
                        if neighbour_right is None and neighbour_bottom is None:
                            embedded = True
                        else:
                            if neighbour_right is not None:
                                if neighbour_right.unit_type != 'bar_unit':
                                    embedded = True
                            if neighbour_bottom is not None:
                                if neighbour_bottom.unit_type != 'bar_unit':
                                    embedded = True
                        if embedded == True:'''
                        self.inputs.append(Input(guiding_text, textbox, is_embedded=True, input_type=3))
                        continue

                # *** 2. A small piece of text at corner of a large Input box ***
                '''if textbox.height / max([c.height for c in textbox.contains]) > 2 and 0 < textbox.containment_area / textbox.area < 0.15 and\
                        min([c.location['left'] for c in textbox.contains]) - textbox.location['left'] > textbox.location['right'] - max([c.location['right'] for c in textbox.contains]):
                    neighbour_top = self.find_neighbour_unit(textbox, 'top',
                                                             connect_bias=self.config.input_compound_recognition_neighbor2_connect_bias,
                                                             align_bias=self.config.input_compound_recognition_neighbor2_align_bias)
                    if neighbour_top is not None and neighbour_top.unit_type == 'text_unit' and neighbour_top.in_input is None and neighbour_top.in_table is None and \
                            textbox.location['top'] - neighbour_top.location['bottom'] < self.config.input_compound_recognition_max_gap_v:
                        textbox.type = 'rectangle'
                        self.inputs.append(Input(neighbour_top, textbox, placeholder=textbox.content))'''

        # *** 3. Normal Input: guide text and input field are separate and aligned ***
        # from left to right
        units = self.sorted_left_unit
        for i, unit in enumerate(units):
            if unit.unit_type == 'text_unit' and unit.in_input is None and unit.in_table is None:
                neighbour_right = self.find_neighbour_unit(unit, direction='right',
                                                           connect_bias=self.config.input_compound_recognition_neighbor3_connect_bias,
                                                           align_bias=self.config.input_compound_recognition_neighbor3_align_bias)
                if neighbour_right is not None and \
                        neighbour_right.unit_type == 'bar_unit' and neighbour_right.type != 'square' and \
                        neighbour_right.in_input is None and neighbour_right.in_table is None and \
                        neighbour_right.location['left'] - unit.location[
                    'right'] < self.config.input_compound_recognition_max_gap_h:
                    self.inputs.append(Input(unit, neighbour_right, input_type=1))
        # from top to bottom
        units = self.sorted_top_unit
        for i, unit in enumerate(units):
            if unit.unit_type == 'text_unit' and unit.in_input is None and unit.in_table is None:
                neighbour_below = self.find_neighbour_unit(unit, direction='below',
                                                           connect_bias=self.config.input_compound_recognition_neighbor3_connect_bias,
                                                           align_bias=self.config.input_compound_recognition_neighbor3_align_bias)
                # units of an input compound with vertical alignment should be left justifying
                if neighbour_below is not None and \
                        neighbour_below.unit_type == 'bar_unit' and neighbour_below.type != 'square' and \
                        neighbour_below.in_input is None and neighbour_below.in_table is None and \
                        neighbour_below.location['top'] - unit.location[
                    'bottom'] < self.config.input_compound_recognition_max_gap_v:
                    # if the bar has text above justified
                    if abs(unit.location['left'] - neighbour_below.location[
                        'left']) < self.config.input_compound_recognition_max_left_justify:
                        self.inputs.append(Input(unit, neighbour_below, input_type=2))
                    # if the bar has no left or right neighbour, then combine it with text above
                    else:
                        bar_left = self.find_neighbour_unit(neighbour_below, direction='left')
                        bar_right = self.find_neighbour_unit(neighbour_below, direction='right')
                        if bar_left is None and bar_right is None:
                            self.inputs.append(Input(unit, neighbour_below, input_type=2))

        for bar in self.bar_units:
            if bar.in_input == None and len(
                    bar.contains) > 0 and bar.in_table == None and bar.is_module_part == False and bar.is_abandoned == False:
                self.inputs.append(Input(bar.contains[0], bar, input_type=3))

    def row_detection(self, unit):
        '''
        Detect row through grouping all left-right connected and justified elements
        :param unit: start unit
        '''
        # if already are detected in a row
        if unit.in_row is not None:
            return unit.in_row
        unit_org = unit

        row = Row(self.row_id)
        self.row_id += 1
        # right forward
        neighbour_right = self.find_neighbour_unit(unit, 'right',
                                                   connect_bias=self.config.row_detection_neighbor_connect_bias,
                                                   align_bias=self.config.row_detection_neighbor_align_bias)
        is_row = False
        # if there is a connected neighbour, add it and the current unit to a Row
        while neighbour_right is not None and unit.is_on_same_line(neighbour_right, 'h',
                                                                   bias_gap=self.config.row_detection_same_line_bias_gap,
                                                                   bias_justify=self.config.row_detection_same_line_bias_justify) \
                and neighbour_right.unit_type == 'bar_unit' and len(neighbour_right.contains) == 0:
            if not is_row:
                row.add_element(unit)
                is_row = True
            # if the neighbour is already in a row, then simply add the current one to the row
            if neighbour_right.in_row is not None:
                row.merge_row(neighbour_right.in_row)
                break
            row.add_element(neighbour_right)
            unit = neighbour_right
            neighbour_right = self.find_neighbour_unit(neighbour_right, 'right',
                                                       connect_bias=self.config.row_detection_neighbor_connect_bias,
                                                       align_bias=self.config.row_detection_neighbor_align_bias)

        # left forward
        unit = unit_org
        neighbour_left = self.find_neighbour_unit(unit, 'left',
                                                  connect_bias=self.config.row_detection_neighbor_connect_bias,
                                                  align_bias=self.config.row_detection_neighbor_align_bias)
        # if there is neighbour on the same row, add it and the current unit to a Row
        while neighbour_left is not None and unit.is_on_same_line(neighbour_left, 'h',
                                                                  bias_gap=self.config.row_detection_same_line_bias_gap,
                                                                  bias_justify=self.config.row_detection_same_line_bias_justify) \
                and neighbour_left.unit_type == 'bar_unit' and len(neighbour_left.contains) == 0:
            if not is_row:
                row.add_element(unit)
                is_row = True
            # if the neighbour is already in a row, then simply add the current one to the row
            if neighbour_left.in_row is not None:
                row.merge_row(neighbour_left.in_row)
                break
            row.add_element(neighbour_left)
            unit = neighbour_left
            neighbour_left = self.find_neighbour_unit(neighbour_left, 'left',
                                                      connect_bias=self.config.row_detection_neighbor_connect_bias,
                                                      align_bias=self.config.row_detection_neighbor_align_bias)

        if len(row.elements) > 1:
            # row.visualize_row(self.img.img.copy(), show=True)
            return row
        else:
            return None

    def detect_table_heading(self, table):
        '''
        Detect heading row for each table
        :param max_gap: max gop between the top row of a table and its top neighbour
        '''
        neighbours = []
        top_row = table.rows[0]
        # record all neighbours above the top row elements
        for ele in top_row.elements:
            n = self.find_neighbour_unit(ele, direction='top',
                                         connect_bias=self.config.detect_table_heading_neighbor_connect_bias,
                                         align_bias=self.config.detect_table_heading_neighbor_align_bias)
            if n is not None and \
                    (n.unit_type == 'text_unit' or (n.unit_type == 'bar_unit' and len(n.contains) > 0)) and abs(
                ele.location['top'] - n.location['bottom']) < self.config.detect_table_heading_max_gap:
                neighbours.append(n)
            else:
                neighbours.append(None)

        heading = Row(row_id=self.row_id)
        self.row_id += 1
        for ele in neighbours:
            if ele is not None:
                heading.add_element(ele)
        if heading.location is not None:
            table.add_heading(heading)

    def detect_table_row_title(self, table):
        titles = []
        t_num = 0
        for row in table.rows:
            first_ele = row.elements[0]
            neighbour = self.find_neighbour_unit(first_ele, direction='left',
                                                 connect_bias=self.config.detect_table_row_title_neighbor_connect_bias,
                                                 align_bias=self.config.detect_table_row_title_neighbor_align_bias)
            if neighbour is not None and (neighbour.unit_type == 'text_unit' or (
                    neighbour.unit_type == 'bar_unit' and len(
                neighbour.contains) > 0)) and neighbour.in_input is None and neighbour.in_table is None:
                titles.append(neighbour)
                t_num += 1
            else:
                titles.append(None)

        if t_num / len(table.rows) >= 0.5:
            for i in range(1, len(titles)):
                if titles[i] is None or titles[i - 1] is None:
                    continue
                if abs(titles[i].location['left'] - titles[i - 1].location[
                    'left']) > self.config.detect_table_row_title_max_title_justify_bias:
                    return
            for i, title in enumerate(titles):
                if title is not None:
                    table.rows[i].row_title = title
                    title.is_module_part = True
                    table.rows[i].add_element(title)
                    table.sort_rows()
                    table.init_bound()

    def table_detection(self):
        '''
        Detect table by detecting continuously matched rows
        '''
        recorded_row_ids = []
        for unit in self.all_units:
            if unit.unit_type == 'bar_unit' and len(unit.contains) == 0:
                # if an element has right(same row) and below(same column) connected elements
                # then check if its row and the row below it are matched
                row = self.row_detection(unit)
                if row is not None:
                    # avoid redundancy
                    if row.row_id in recorded_row_ids:
                        continue
                    else:
                        recorded_row_ids.append(row.row_id)

                    if row.parent_table is not None:
                        continue
                    else:
                        table = Table(self.table_id)
                        self.table_id += 1

                    # *** detect down forwards ***
                    unit_a = unit
                    unit_b = self.find_neighbour_unit(unit_a, 'below',
                                                      connect_bias=self.config.table_detection_neighbor_connect_bias,
                                                      align_bias=self.config.table_detection_neighbor_align_bias)
                    if unit_b is None or unit_b.unit_type != 'bar_unit' or len(unit_b.contains) > 0:
                        continue
                    row_a = row
                    # check if the unit has neighbour on the same colunm
                    while unit_b is not None and unit_a.is_on_same_line(ele_b=unit_b, direction='v',
                                                                        bias_gap=self.config.table_detection_same_line_bias_gap,
                                                                        bias_justify=self.config.table_detection_same_line_bias_justify):
                        row_b = self.row_detection(unit_b)
                        # check if its row and the row below it matches
                        # merge matched parts of the two rows to a table
                        if row_b is not None and row_a.is_matched(row_b):
                            if row_b.parent_table is not None:
                                table.merge_table(row_b.parent_table)
                            else:
                                if table.is_empty():
                                    table.add_rows([row_a, row_b])
                                else:
                                    table.add_row(row_b)
                            unit_a = unit_b
                            row_a = row_b
                            unit_b = self.find_neighbour_unit(unit_a, 'below',
                                                              connect_bias=self.config.table_detection_neighbor_connect_bias,
                                                              align_bias=self.config.table_detection_neighbor_align_bias)
                        else:
                            break

                    # *** detect up forwards ***
                    unit_a = unit
                    unit_b = self.find_neighbour_unit(unit_a, 'top',
                                                      connect_bias=self.config.table_detection_neighbor_connect_bias,
                                                      align_bias=self.config.table_detection_neighbor_align_bias)
                    if unit_b is None or unit_b.unit_type != 'bar_unit' or len(unit_b.contains) > 0:
                        continue
                    row_a = row
                    # check if the unit has neighbour on the same colunm
                    while unit_b is not None and unit_a.is_on_same_line(unit_b, direction='v',
                                                                        bias_gap=self.config.table_detection_same_line_bias_gap,
                                                                        bias_justify=self.config.table_detection_same_line_bias_justify):
                        row_b = self.row_detection(unit_b)
                        # check if its row and the row below it matches
                        # merge matched parts of the two rows to a table
                        if row_b is not None and row_a.is_matched(row_b):
                            if row_b.parent_table is not None:
                                table.merge_table(row_b.parent_table)
                            else:
                                if table.is_empty():
                                    table.add_rows([row_a, row_b])
                                else:
                                    table.add_row(row_b)
                            unit_a = unit_b
                            row_a = row_b
                            unit_b = self.find_neighbour_unit(unit_a, 'top',
                                                              connect_bias=self.config.table_detection_neighbor_connect_bias,
                                                              align_bias=self.config.table_detection_neighbor_align_bias)
                        else:
                            break

                    if not table.is_empty():
                        # board = self.get_img_copy()
                        # table.visualize_table(board)
                        # unit.visualize_element(board, color=(0,0,255), show=True)
                        self.tables.append(table)

        return self.tables

    '''
    *************************
    *** Module Refinement ***
    *************************
    '''

    def table_merge_contained_eles(self, table):
        '''
        Merge elements that are not grouped but contained in the table
        '''
        for unit in self.all_units:
            if unit.in_row is not None or unit.in_table is not None:
                continue
            if table.is_ele_contained_in_table(unit, bias=self.config.is_ele_contained_in_table_bias):
                table.insert_element(unit, insert_bias=self.config.insert_element_alignment_bias)

    def table_refine(self):
        # *** Step 1. Merge elements that are not grouped but contained in a table ***
        for table in self.tables:
            self.table_merge_contained_eles(table)

        # *** Step 2. Heading detection for table ***
        for table in self.tables:
            self.detect_table_heading(table)
            self.table_merge_contained_eles(table)
            table.merge_vertical_texts_in_cell(self.config.row_merge_vertical_texts_in_cell_alignment)

        # *** Step 3. Split columns of a table according to the heading ***
        for table in self.tables:
            table.split_columns(self.config)

        # *** Step 4. Detect row title for each row ***
        for table in self.tables:
            self.detect_table_row_title(table)
            self.table_merge_contained_eles(table)
            table.merge_vertical_texts_in_cell(self.config.row_merge_vertical_texts_in_cell_alignment)

        # *** Step 5. Remove noises according to column ***
        for table in self.tables:
            table.rm_noisy_element()

    def input_refine(self):
        self.inputs = sorted(self.inputs, key=lambda x: x.location['left'])
        for ipt in self.inputs:
            # skip inputs where its guide text and input filed in the same box
            '''if ipt.is_embedded:
                continue'''
            # merge intersected text into guide_text
            changed = True
            while changed:
                changed = False
                for text in self.text_units:
                    if not text.is_module_part and not text.is_abandoned and \
                            ipt.guide_text.unit_group_id == text.unit_group_id and \
                            (ipt.guide_text.pos_relation(text,
                                                         self.config.input_refine_pos_relation_bias) != 0 or ipt.pos_relation(
                                text, self.config.input_refine_pos_relation_bias) != 0):
                        ipt.merge_guide_text(text)
                        changed = True
                        break

            # merged connected input field horizontally
            changed = True
            while changed:
                changed = False
                self.bar_units = sorted(self.bar_units, key=lambda x: x.location['left'])
                for bar in self.bar_units:
                    if not bar.is_module_part and not bar.is_abandoned and bar.type == 'rectangle' and \
                            ipt.is_connected_field(bar, direction='h',
                                                   bias_gap=self.config.input_refine_is_connected_bias_gap,
                                                   bias_justify=self.config.input_refine_is_connected_bias_justify):
                        ipt.merge_input_field(bar)
                        ipt.input_type = 1
                        changed = True
                        break

        # merged connected input field vertically
        self.inputs = sorted(self.inputs, key=lambda x: x.location['top'])
        for ipt in self.inputs:
            changed = True
            while changed:
                changed = False
                self.bar_units = sorted(self.bar_units, key=lambda x: x.location['top'])
                for bar in self.bar_units:
                    if not bar.is_module_part and not bar.is_abandoned and bar.type == 'rectangle' and \
                            ipt.is_connected_field(bar, direction='v',
                                                   bias_gap=self.config.input_refine_is_connected_bias_gap,
                                                   bias_justify=self.config.input_refine_is_connected_bias_justify):
                        ipt.merge_input_field(bar)
                        ipt.input_type = 2
                        changed = True
                        break

    def text_refine(self):
        '''
        Merge intersected ungrouped texts
        '''
        texts = []
        others = []
        for text in self.texts:
            if not text.in_box and not text.is_abandoned and not text.is_module_part:
                texts.append(text)
            else:
                others.append(text)

        changed = True
        while changed:
            changed = False
            temp_set = []
            for text_a in texts:
                merged = False
                for text_b in temp_set:
                    if text_a.pos_relation(text_b, self.config.text_refine_pos_relation_bias) != 0:
                        text_b.merge_text(text_a, direction='v')
                        merged = True
                        changed = True
                        break
                if not merged:
                    temp_set.append(text_a)
            texts = temp_set.copy()

        self.texts = texts + others

    '''
    *********************
    *** Visualization ***
    *********************
    '''

    def get_img_copy(self):
        return self.img.img.copy()

    def visualize_vertical_separators(self):
        if self.vertical_separators is None:
            return
        board = self.get_img_copy()
        for separator in self.vertical_separators:
            cv2.rectangle(board, (separator['left'], separator['top']), (separator['right'], separator['bottom']),
                          (0, 255, 0), 1)
        cv2.imshow('v-separators', board)
        cv2.waitKey()
        cv2.destroyWindow('v-separators')

    def visualize_unit_groups(self):
        for group in self.unit_groups:
            for g in group:
                board = self.get_img_copy()
                for u in g:
                    u.visualize_element(board)
                cv2.imshow('groups', board)
                cv2.waitKey()
                cv2.destroyAllWindows()

    def visualize_all_elements(self):
        board = self.get_img_copy()
        for text in self.texts:
            # if not text.in_box and not text.is_abandoned:
            text.visualize_element(board)

        for rec in self.rectangles:
            # if not rec.is_abandoned:
            rec.visualize_element(board)

        for squ in self.squares:
            # if not squ.is_abandoned:
            squ.visualize_element(board)

        for line in self.lines:
            # if not line.is_abandoned:
            line.visualize_element(board)

        for table in self.tables:
            table.visualize_element(board, color=(255, 255, 0))

        cv2.namedWindow('form', cv2.WINDOW_NORMAL);
        cv2.imshow('form', board)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def visualize_table(self):
        board = self.get_img_copy()
        for table in self.tables:
            table.visualize_element(board, color=(255, 255, 0))

        cv2.imshow('form', board)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def visualize_all_elements_one_by_one(self, reset=False, opt='org', text_on=False, rec_on=False, squ_on=False,
                                          line_on=False, table_on=False):
        if opt == 'binary':
            board_origin = np.zeros((self.img.img_shape[0], self.img.img_shape[1], 3))
        if opt == 'org':
            board_origin = self.get_img_copy()
        board = board_origin.copy()
        for text in self.texts:
            # if not text.in_box and n ot text.is_abandoned:
            if reset == True:
                board = board_origin.copy()
            # text.visualize_element(board)
            if text_on == True:
                cv2.namedWindow('form', cv2.WINDOW_NORMAL);
                cv2.imshow('form', board)
                cv2.waitKey()
                cv2.destroyAllWindows()

        for rec in self.rectangles:
            # if not rec.is_abandoned:
            if reset == True:
                board = board_origin.copy()
            rec.visualize_element(board)
            if rec_on == True:
                cv2.namedWindow('form', cv2.WINDOW_NORMAL);
                cv2.imshow('form', board)
                cv2.waitKey()
                cv2.destroyAllWindows()

        for squ in self.squares:
            # if not squ.is_abandoned:
            if reset == True:
                board = board_origin.copy()
            squ.visualize_element(board)
            if squ_on == True:
                cv2.namedWindow('form', cv2.WINDOW_NORMAL);
                cv2.imshow('form', board)
                cv2.waitKey()
                cv2.destroyAllWindows()

        for line in self.lines:
            # if not line.is_abandoned:
            if reset == True:
                board = board_origin.copy()
            line.visualize_element(board)
            if line_on == True:
                cv2.namedWindow('form', cv2.WINDOW_NORMAL);
                cv2.imshow('form', board)
                cv2.waitKey()
                cv2.destroyAllWindows()

        for table in self.tables:
            # table.visualize_element(board, color=(255,255,0))
            if reset == True:
                board = board_origin.copy()
            if table_on == True:
                cv2.namedWindow('form', cv2.WINDOW_NORMAL);
                cv2.imshow('form', board)
                cv2.waitKey()
                cv2.destroyAllWindows()


    def visualize_units(self):
        board = self.get_img_copy()
        for text_unit in self.text_units:
            text_unit.visualize_element(board, color=(255, 0, 0))
        for bar_unit in self.bar_units:
            bar_unit.visualize_element(board, color=(0, 255, 0))
        cv2.namedWindow('Units', cv2.WINDOW_NORMAL);
        cv2.imshow('Units', board)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def visualize_inputs(self):
        board = self.get_img_copy()
        for ipt in self.inputs:
            if ipt.input_type == 1:
                ipt.visualize_element(board, color=(255, 0, 0), line=2)
            elif ipt.input_type == 2:
                ipt.visualize_element(board, color=(0, 255, 0), line=2)
            elif ipt.input_type == 3:
                ipt.visualize_element(board, color=(0, 0, 255), line=2)
            if ipt.input_type == 4:
                ipt.visualize_element(board, color=(255, 255, 0), line=2)
        cv2.imshow('Input', board)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def visualize_inputs_and_tables(self):
        board = self.get_img_copy()
        for ipt in self.inputs:
            if ipt.input_type == 1:
                ipt.visualize_element(board, color=(255, 0, 0), line=2)
            elif ipt.input_type == 2:
                ipt.visualize_element(board, color=(0, 255, 0), line=2)
            elif ipt.input_type == 3:
                ipt.visualize_element(board, color=(0, 0, 255), line=2)
            if ipt.input_type == 4:
                ipt.visualize_element(board, color=(255, 255, 0), line=2)
        for table in self.tables:
            table.visualize_element(board, color=(0, 255, 255))
        cv2.imshow('Input_and_table', board)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def export_visualize_inputs_and_tables(self, export_dir):
        board = self.get_img_copy()
        for ipt in self.inputs:
            if ipt.input_type == 1:
                ipt.visualize_element(board, color=(255, 0, 0), line=2)  # blue
            elif ipt.input_type == 2:
                ipt.visualize_element(board, color=(0, 255, 0), line=2)  # green
            elif ipt.input_type == 3:
                ipt.visualize_element(board, color=(0, 0, 255), line=2)  # red
            if ipt.input_type == 4:
                ipt.visualize_element(board, color=(255, 255, 0), line=2)  #
        for table in self.tables:
            table.visualize_element(board, color=(0, 255, 255))  # yellow

        if export_dir is None:
            print("No correct path")
        print('Input detection results write to:', os.path.join(export_dir, self.form_name + '.jpg'))
        cv2.imwrite(os.path.join(export_dir, self.form_name + '.jpg'), board)

    def visualize_inputs_one_by_one(self):
        for ipt in self.inputs:
            board = self.get_img_copy()
            ipt.visualize_element(board, color=(255, 0, 255), line=2)
            # ipt.visualize_input_overlay(board)
            cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
            cv2.imshow('Input', board)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def visualize_detection_result(self):
        board = self.get_img_copy()
        for text in self.texts:
            if not text.in_box and not text.is_abandoned and not text.is_module_part:
                text.visualize_element(board)

        for rec in self.rectangles:
            if not rec.is_abandoned and not rec.is_module_part:
                rec.visualize_element(board)

        for line in self.lines:
            if not line.is_abandoned and not line.is_module_part:
                line.visualize_element(board)

        for table in self.tables:
            table.visualize_element(board, color=(255, 255, 0))

        for ipt in self.inputs:
            ipt.visualize_element(board, color=(255, 0, 255))

        self.detection_result_img = board
        cv2.imshow('form', board)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def visualize_detection_result_one_by_one(self, reset=False,
                                              opt='org', text_test=False, rec_test=False,
                                              line_test=False,
                                              table_test=False, input_test=False):
        if opt == 'org':
            board_org = self.get_img_copy()
        else:
            board_org = np.zeros((self.img.img_shape[0], self.img.img_shape[1], 3))
        board = board_org.copy()
        for text in self.texts:
            if not text.in_box and not text.is_abandoned and not text.is_module_part:
                if reset == True:
                    board = board_org.copy()
                # text.visualize_element(board)
                if text_test == True:
                    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                    cv2.imshow('result', board)
                    cv2.waitKey()
                    cv2.destroyAllWindows()

        for rec in self.rectangles:
            if not rec.is_abandoned and not rec.is_module_part:
                if reset == True:
                    board = board_org.copy()
                rec.visualize_element(board)
                if text_test == True:
                    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                    cv2.imshow('result', board)
                    cv2.waitKey()
                    cv2.destroyAllWindows()

        for line in self.lines:
            if not line.is_abandoned and not line.is_module_part:
                if reset == True:
                    board = board_org.copy()
                line.visualize_element(board)
                if text_test == True:
                    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                    cv2.imshow('result', board)
                    cv2.waitKey()
                    cv2.destroyAllWindows()

        for table in self.tables:
            if reset == True:
                board = board_org.copy()
            table.visualize_element(board, color=(255, 255, 0))
            if text_test == True:
                cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                cv2.imshow('result', board)
                cv2.waitKey()
                cv2.destroyAllWindows()

        for ipt in self.inputs:
            if reset == True:
                board = board_org.copy()
            ipt.visualize_element(board, color=(255, 0, 255))
            if text_test == True:
                cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                cv2.imshow('result', board)
                cv2.waitKey()
                cv2.destroyAllWindows()

        self.detection_result_img = board
        cv2.imshow('form', board)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def export_detection_result_json(self, export_dir=None):
        detection_text = []
        detection_rec = []
        detection_squ = []
        detection_line = []
        detection_table = []
        detection_ipt = []

        for text in self.texts:
            if not text.in_box and not text.is_abandoned and not text.is_module_part:
                text_json = {'type': text.type,
                             'content': text.content,
                             'location': text.location}
                detection_text.append(text_json)

        for rec in self.rectangles:
            if not rec.is_abandoned and not rec.is_module_part:
                rec_json = {'type': rec.type,
                            'unit_type': rec.unit_type,
                            'location': rec.location}
                detection_rec.append(rec_json)

        for squ in self.squares:
            if not squ.is_abandoned and not squ.is_module_part:
                squ_json = {'type': squ.type,
                            'unit_type': squ.unit_type,
                            'location': squ.location}
                detection_squ.append(squ_json)

        for line in self.lines:
            if not line.is_abandoned and not line.is_module_part:
                line_json = {'type': line.type,
                             'unit_type': line.unit_type,
                             'location': line.location}
                detection_line.append(line_json)

        for table in self.tables:
            row_dic_list = []
            for row in table.rows:
                row_json = {'type': row.type,
                            'location': row.location}
                row_dic_list.append(row_json)
            table_json = {'type': table.type,
                          'location': table.location,
                          'rows': row_dic_list}
            detection_table.append(table_json)

        for ipt in self.inputs:
            if ipt.guide_text.type == 'textbox':
                guide_text = ipt.guide_text.contains[0].content
            elif ipt.guide_text.type == 'text':
                guide_text = ipt.guide_text.content
            else:
                guide_text = 'None'
            input_field_list = []
            for ipt_field in ipt.input_fields:
                field_json = {'type': ipt_field.type,
                              'location': ipt_field.location}
                input_field_list.append(field_json)
            ipt_json = {'type': ipt.type,
                        'input_type': ipt.input_type,
                        'location': ipt.location,
                        'guide_text': guide_text,
                        'input_fields': input_field_list}
            detection_ipt.append(ipt_json)

        detection_result_json = {'texts': detection_text,
                                 'rectangles': detection_rec,
                                 'squares': detection_squ,
                                 'lines': detection_line,
                                 'tables': detection_table,
                                 'inputs': detection_ipt}
        # structure:
        # {'texts':[text1, text2, ...],                 text = {'type', 'content', 'location'}
        #  'rectangles':[rec1, rec2, ...],              rec = {'type', 'unit_type', 'location'}
        #  'squares':[squ1, squ2, ...],                 squ = {'type', 'unit_type', 'location'}
        #  'lines':[line1, line2, ...],                 line = {'type', 'unit_type', 'location'}
        #  'tables':[table1, table2, ...],              table = {'type', 'location', 'rows' = [...}]}  row = {'type', 'location'}
        #  'inputs':[input1, input2, ...]}              input = {'type', 'location', 'guide_text', 'input_fields' = [...]}  input_field = {'type', 'location'}
        json.dump(detection_result_json, open(os.path.join(export_dir, self.form_name + '.json'), 'w+'), indent=4)
        print('Result JSON Write to:', os.path.join(export_dir, self.form_name + '.json'))

    def export_detection_result_img(self, export_dir=None):
        board = self.get_img_copy()
        for text in self.texts:
            if not text.in_box and not text.is_abandoned and not text.is_module_part:
                text.visualize_element(board)

        for rec in self.rectangles:
            if not rec.is_abandoned and not rec.is_module_part:
                rec.visualize_element(board)

        for line in self.lines:
            if not line.is_abandoned and not line.is_module_part:
                line.visualize_element(board)

        for table in self.tables:
            table.visualize_element(board, color=(0, 0, 255))

        for ipt in self.inputs:
            ipt.visualize_element(board, color=(255, 0, 255))

        self.detection_result_img = board
        if export_dir is None:
            export_dir = self.export_dir
        print('Write to:', os.path.join(export_dir, self.form_name + '.jpg'))
        cv2.imwrite(os.path.join(export_dir, self.form_name + '.jpg'), board)
