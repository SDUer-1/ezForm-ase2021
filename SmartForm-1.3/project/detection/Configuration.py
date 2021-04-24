
class Configuration:
    def __init__(self):
        self.element_refine_bias = 5
        self.sentences_recognition_manually_set = False
        self.sentences_recognition_bias_gap = 4
        self.sentences_recognition_bias_justify = 4
        self.shrink_text_and_filter_noises = 3
        self.get_elements_min_area = 20                     #Image line74
        #Element.is_rectangle_or_square
        self.is_line_max_thickness = 3                      #Element line77
        self.is_line_min_length = 20                        #Element line77
        self.filter_detection_noises_pos_relation_bias = 0  #Form line 471
        self.border_and_textbox_recognition_pos_relation_bias = 0  # Form line 471
        self.is_textbox_or_border_ratio = 0.6               #Element line 328
        self.textbox_merge_alignment_bias = 0               #Element line 348
        self.textbox_merge_v_max_merged_gap = 20            #Element line 348
        self.same_character_box_updownside_bias = 10        #Element line 306
        self.same_character_box_gap = 10                    #Element line 306
        self.same_character_box_pos_relation_bias = 0  # Form line 471
        self.same_character_box_manually_set = False        #Element line 306
        self.border_line_recognition_left_bias = 5          #Form line545
        self.border_line_recognition_top_bias = 10          #Form line545
        self.border_line_recognition_neighbor_connect_bias = 10  # Form line545
        self.border_line_recognition_neighbor_align_bias = 4  # Form line545
        self.row_detection_neighbor_connect_bias = 10       # Form line680
        self.row_detection_neighbor_align_bias = 4          # Form line680
        self.row_detection_same_line_bias_gap = 10          # Form line680
        self.row_detection_same_line_bias_justify = 4       # Form line680
        self.table_detection_neighbor_connect_bias = 10     # Form line794
        self.table_detection_neighbor_align_bias = 4        # Form line794
        self.table_detection_same_line_bias_gap = 4         # Form line794
        self.table_detection_same_line_bias_justify = 4     # Form line794
        self.is_ele_contained_in_table_bias = 4             # Table line100
        self.insert_element_alignment_bias = 1              # Table line117
        self.detect_table_heading_neighbor_connect_bias = 10    # Form line744
        self.detect_table_heading_neighbor_align_bias = 4       # Form line744
        self.detect_table_heading_max_gap = 20              # Form line744
        self.row_merge_vertical_texts_in_cell_alignment = 4 # Row line140
        self.split_col_max_bias_justify = 10                # Table line123
        self.detect_table_row_title_neighbor_connect_bias = 10  # Form line770
        self.detect_table_row_title_neighbor_align_bias = 4 # Form line770
        self.detect_table_row_title_max_title_justify_bias = 10  # Form line770
        self.input_compound_recognition_max_gap_h = 150     # Form line601
        self.input_compound_recognition_max_gap_v = 30      # Form line601
        self.input_compound_recognition_max_left_justify = 8 # Form line601
        self.input_compound_recognition_neighbor2_connect_bias = 10  # Form line601
        self.input_compound_recognition_neighbor2_align_bias = 4  # Form line601
        self.input_compound_recognition_neighbor3_connect_bias = 10  # Form line601
        self.input_compound_recognition_neighbor3_align_bias = 10  # Form line601
        self.input_compound_recognition_neighbor4_connect_bias = 10  # Form line601
        self.input_compound_recognition_neighbor4_align_bias = 4  # Form line601
        self.input_refine_pos_relation_bias = 0              # Form line942
        self.input_refine_is_connected_bias_gap = 10
        self.input_refine_is_connected_bias_justify = 10
        self.text_refine_pos_relation_bias = 0              # Form line1005


    def set_sentences_recognition_parameter(self,text_a,text_b):
        if self.sentences_recognition_manually_set == False:
            self.sentences_recognition_bias_justify = 0.6 * max(text_a.height,text_b.height)
            self.sentences_recognition_bias_gap = 1.5 * max((text_a.width / len(text_a.content)),
                                                               (text_b.width / len(text_b.content)))

    def set_same_character_box_parameter(self,ele_a,ele_b):
        if ele_a.location['left'] < ele_b.location['left']:
            ele_l = ele_a
            ele_r = ele_b
        else:
            ele_l = ele_b
            ele_r = ele_a
        if self.same_character_box_manually_set == False:
            self.same_character_box_updownside_bias = 0.2 * ele_l.height
            self.sentences_recognition_bias_gap = 0.8 * max(ele_l.width / ele_l.character_num,ele_r.width / ele_r.character_num)

    def set_split_col_max_bias_justify(self,head):
        self.split_col_max_bias_justify = int(head.width / 2)