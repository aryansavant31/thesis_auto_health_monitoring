
# ------------- AMMF ----------------

def ammf_acc_faults(level):
    # ammf faults (l2 medium)
    fault_l2_1 = [get_augment_config('glitch', prob=0.07, std_fac=2, add_next=True),
                    get_augment_config('sine', freqs=[3, 2], std_facs=[1.8, 2])]
    fault_l2_2 = [get_augment_config('glitch', prob=0.08, std_fac=1.8, add_next=True),
                    get_augment_config('sine', freqs=[2, 4], std_facs=[1.8, 1.5])]
    fault_l2_3 = [get_augment_config('glitch', prob=0.09, std_fac=2.2, add_next=True),
                    get_augment_config('sine', freqs=[2, 4, 5], std_facs=[2, 1.8, 1.5])]
    fault_l2_4 = [get_augment_config('glitch', prob=0.03, std_fac=1.9, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 3], std_facs=[1.7, 1.9, 2])]
    fault_l2_5 = [get_augment_config('glitch', prob=0.02, std_fac=1.5, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 4], std_facs=[1.8, 2, 1.9])]
    fault_l2_6 = [get_augment_config('glitch', prob=0.05, std_fac=1.9, add_next=True),
                    get_augment_config('sine', freqs=[1, 3, 5, 7], std_facs=[1.6, 2.1, 1.8, 1.9]),]
    
    all_l2_faults = [fault_l2_1, fault_l2_2, fault_l2_3, fault_l2_4, fault_l2_5, fault_l2_6, fault_l2_1, fault_l2_3]
    
    # ammf faults (l3 subtle)
    fault_l3_1 = [get_augment_config('glitch', prob=0.07, std_fac=1.2, add_next=True),
                    get_augment_config('sine', freqs=[3, 2], std_facs=[1.3, 1.4])]
    fault_l3_2 = [get_augment_config('glitch', prob=0.08, std_fac=1.5, add_next=True),
                    get_augment_config('sine', freqs=[2, 4], std_facs=[1.2, 1.3])]
    fault_l3_3 = [get_augment_config('glitch', prob=0.09, std_fac=1.35, add_next=True),
                    get_augment_config('sine', freqs=[2, 4, 5], std_facs=[1.2, 1.5, 1])]
    fault_l3_4 = [get_augment_config('glitch', prob=0.03, std_fac=1.25, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 3], std_facs=[1.3, 1.4, 1.25])]
    fault_l3_5 = [get_augment_config('glitch', prob=0.02, std_fac=1.2, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 4], std_facs=[1, 1.4, 1.1])]
    fault_l3_6 = [get_augment_config('glitch', prob=0.05, std_fac=1.3, add_next=True),
                    get_augment_config('sine', freqs=[1, 3, 5, 7], std_facs=[1.2, 1.1, 1, 1.3]),]
    
    all_l3_faults = [fault_l3_1, fault_l3_2, fault_l3_3, fault_l3_4, fault_l3_5, fault_l3_6, fault_l3_3, fault_l3_4]

    # ammf faults (l4 subtle)
    fault_l4_1 = [get_augment_config('glitch', prob=0.07, std_fac=0.5, add_next=True),
                    get_augment_config('sine', freqs=[3, 2], std_facs=[0.65, 0.7])]
    fault_l4_2 = [get_augment_config('glitch', prob=0.08, std_fac=0.55, add_next=True),
                    get_augment_config('sine', freqs=[2, 4], std_facs=[0.8, 0.6])]
    fault_l4_3 = [get_augment_config('glitch', prob=0.09, std_fac=0.75, add_next=True),
                    get_augment_config('sine', freqs=[2, 4, 5], std_facs=[0.9, 0.8, 0.7])]
    fault_l4_4 = [get_augment_config('glitch', prob=0.03, std_fac=0.67, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 3], std_facs=[0.88, 0.65, 0.55])]
    fault_l4_5 = [get_augment_config('glitch', prob=0.02, std_fac=0.5, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 4], std_facs=[0.6, 0.7, 0.5])]
    fault_l4_6 = [get_augment_config('glitch', prob=0.05, std_fac=0.67, add_next=True),
                    get_augment_config('sine', freqs=[1, 3, 5, 7], std_facs=[0.7, 0.85, 0.6, 0.5]),]
    
    all_l4_faults = [fault_l4_1, fault_l4_2, fault_l4_3, fault_l4_4, fault_l4_5, fault_l4_6, fault_l4_4, fault_l4_5]

    if level == 2:
        return all_l2_faults
    elif level == 3:
        return all_l3_faults
    elif level == 4:
        return all_l4_faults
    
def ammf_ctrl_faults(level):
    fault_l2_1 = [get_augment_config('glitch', prob=0.07, std_fac=2, add_next=True),
                    get_augment_config('sine', freqs=[30, 200, 500], std_facs=[1.8, 2, 1.5])]
    fault_l2_2 = [get_augment_config('glitch', prob=0.08, std_fac=1.8, add_next=True),
                    get_augment_config('sine', freqs=[56, 100], std_facs=[1.8, 1.5])]
    fault_l2_3 = [get_augment_config('glitch', prob=0.09, std_fac=2.2, add_next=True),
                    get_augment_config('sine', freqs=[150, 405, 556], std_facs=[2, 1.8, 1.5])]
    fault_l2_4 = [get_augment_config('glitch', prob=0.03, std_fac=1.9, add_next=True),
                    get_augment_config('sine', freqs=[100, 235, 375], std_facs=[1.7, 1.9, 2])]
    fault_l2_5 = [get_augment_config('glitch', prob=0.02, std_fac=1.5, add_next=True),
                    get_augment_config('sine', freqs=[156, 220, 400], std_facs=[1.8, 2, 1.9])]
    fault_l2_6 = [get_augment_config('glitch', prob=0.05, std_fac=1.9, add_next=True),
                    get_augment_config('sine', freqs=[45, 300, 550, 700], std_facs=[1.6, 2.1, 1.3, 0.8]),]
    
    all_l2_faults = [fault_l2_1, fault_l2_2, fault_l2_3, fault_l2_4, fault_l2_5, fault_l2_6, fault_l2_1, fault_l2_2]

    # l3 level
    fault_l3_1 = [get_augment_config('glitch', prob=0.07, std_fac=0.7, add_next=True),
                    get_augment_config('sine', freqs=[30, 200, 500], std_facs=[0.7, 0.9, 0.6])]
    fault_l3_2 = [get_augment_config('glitch', prob=0.08, std_fac=0.75, add_next=True),
                    get_augment_config('sine', freqs=[56, 100], std_facs=[0.7, 0.9])]
    fault_l3_3 = [get_augment_config('glitch', prob=0.09, std_fac=0.6, add_next=True),
                    get_augment_config('sine', freqs=[150, 405, 556], std_facs=[0.9, 1, 0.77])]
    fault_l3_4 = [get_augment_config('glitch', prob=0.03, std_fac=0.7, add_next=True),
                    get_augment_config('sine', freqs=[100, 235, 375], std_facs=[0.7, 0.95, 0.8])]
    fault_l3_5 = [get_augment_config('glitch', prob=0.02, std_fac=0.9, add_next=True),
                    get_augment_config('sine', freqs=[156, 220, 400], std_facs=[0.6, 0.7, 0.9])]
    fault_l3_6 = [get_augment_config('glitch', prob=0.05, std_fac=0.7, add_next=True),
                    get_augment_config('sine', freqs=[45, 300, 550, 700], std_facs=[0.9, 0.77, 0.6, 0.5]),]
    
    all_l3_faults = [fault_l3_1, fault_l3_2, fault_l3_3, fault_l3_4, fault_l3_5, fault_l3_6, fault_l3_1, fault_l3_2]

    # l4 level
    fault_l4_1 = [get_augment_config('glitch', prob=0.07, std_fac=0.5, add_next=True),
                    get_augment_config('sine', freqs=[30, 200, 500], std_facs=[0.5, 0.4, 0.4])]
    fault_l4_2 = [get_augment_config('glitch', prob=0.08, std_fac=0.55, add_next=True),
                    get_augment_config('sine', freqs=[56, 100], std_facs=[0.5, 0.7])]
    fault_l4_3 = [get_augment_config('glitch', prob=0.09, std_fac=0.4, add_next=True),
                    get_augment_config('sine', freqs=[150, 405, 556], std_facs=[0.7, 0.8, 0.55])]
    fault_l4_4 = [get_augment_config('glitch', prob=0.03, std_fac=0.4, add_next=True),
                    get_augment_config('sine', freqs=[100, 235, 375], std_facs=[0.5, 0.75, 0.6])]
    fault_l4_5 = [get_augment_config('glitch', prob=0.02, std_fac=0.7, add_next=True),
                    get_augment_config('sine', freqs=[156, 220, 400], std_facs=[0.4, 0.5, 0.7])]
    fault_l4_6 = [get_augment_config('glitch', prob=0.05, std_fac=0.5, add_next=True),
                    get_augment_config('sine', freqs=[45, 300, 550, 700], std_facs=[0.5, 0.55, 0.7, 0.3]),]
    
    all_l4_faults = [fault_l4_1, fault_l4_2, fault_l4_3, fault_l4_4, fault_l4_5, fault_l4_6, fault_l4_1, fault_l4_2]

    if level == 2:
        return all_l2_faults
    elif level == 3:
        return all_l3_faults
    elif level == 4:
        return all_l4_faults

# ----------- POB_LS1 ---------------
    
def pob_ls_ctrl_faults(level):
    # ctrl faults (l2 medium)
    fault_l2_1 = [get_augment_config('glitch', prob=0.07, std_fac=2, add_next=True),
                    get_augment_config('sine', freqs=[30, 200, 500], std_facs=[1.8, 2, 1.5])]
    fault_l2_2 = [get_augment_config('glitch', prob=0.08, std_fac=1.8, add_next=True),
                    get_augment_config('sine', freqs=[56, 100], std_facs=[1.8, 1.5])]
    fault_l2_3 = [get_augment_config('glitch', prob=0.09, std_fac=2.2, add_next=True),
                    get_augment_config('sine', freqs=[150, 405, 556], std_facs=[2, 1.8, 1.5])]
    fault_l2_4 = [get_augment_config('glitch', prob=0.03, std_fac=1.9, add_next=True),
                    get_augment_config('sine', freqs=[100, 235, 375], std_facs=[1.7, 1.9, 2])]
    fault_l2_5 = [get_augment_config('glitch', prob=0.02, std_fac=1.5, add_next=True),
                    get_augment_config('sine', freqs=[156, 220, 400], std_facs=[1.8, 2, 1.9])]
    fault_l2_6 = [get_augment_config('glitch', prob=0.05, std_fac=1.9, add_next=True),
                    get_augment_config('sine', freqs=[45, 300, 550, 700], std_facs=[1.6, 2.1, 1.3, 0.8]),]
    
    all_l2_faults = [fault_l2_1, fault_l2_2, fault_l2_3, fault_l2_4, fault_l2_5, fault_l2_6, fault_l2_1, fault_l2_2]
    
    # ctrl faults (l4 subtler)
    fault_l4_1 = [get_augment_config('glitch', prob=0.07, std_fac=0.5, add_next=True),
                    get_augment_config('sine', freqs=[30, 200, 500], std_facs=[0.5, 0.4, 0.4])]
    fault_l4_2 = [get_augment_config('glitch', prob=0.08, std_fac=0.55, add_next=True),
                    get_augment_config('sine', freqs=[56, 100], std_facs=[0.5, 0.7])]
    fault_l4_3 = [get_augment_config('glitch', prob=0.09, std_fac=0.4, add_next=True),
                    get_augment_config('sine', freqs=[150, 405, 556], std_facs=[0.7, 0.8, 0.55])]
    fault_l4_4 = [get_augment_config('glitch', prob=0.03, std_fac=0.4, add_next=True),
                    get_augment_config('sine', freqs=[100, 235, 375], std_facs=[0.5, 0.75, 0.6])]
    fault_l4_5 = [get_augment_config('glitch', prob=0.02, std_fac=0.7, add_next=True),
                    get_augment_config('sine', freqs=[156, 220, 400], std_facs=[0.4, 0.5, 0.7])]
    fault_l4_6 = [get_augment_config('glitch', prob=0.05, std_fac=0.5, add_next=True),
                    get_augment_config('sine', freqs=[45, 300, 550, 700], std_facs=[1.3, 1.2, 1, 0.7]),] # l3
    
    all_l4_faults = [fault_l4_1, fault_l4_2, fault_l4_3, fault_l4_4, fault_l4_5, fault_l4_6, fault_l4_1, fault_l4_2]

    if level == 2:
        return all_l2_faults
    elif level == 4:
        return all_l4_faults
    

# ----------- WS_LS1 ----------------

def ws_ls1_ctrl_faults(level):
    # ctrl faults (l2 medium)
    fault_l2_1 = [get_augment_config('glitch', prob=0.07, std_fac=2, add_next=True),
                    get_augment_config('sine', freqs=[30, 200, 500], std_facs=[1.8, 2, 1.5])]
    fault_l2_2 = [get_augment_config('glitch', prob=0.08, std_fac=1.8, add_next=True),
                    get_augment_config('sine', freqs=[56, 100], std_facs=[1.8, 1.5])]
    fault_l2_3 = [get_augment_config('glitch', prob=0.09, std_fac=2.2, add_next=True),
                    get_augment_config('sine', freqs=[150, 405, 556], std_facs=[2, 1.8, 1.5])]
    fault_l2_4 = [get_augment_config('glitch', prob=0.03, std_fac=1.9, add_next=True),
                    get_augment_config('sine', freqs=[100, 235, 375], std_facs=[1.7, 1.9, 2])]
    fault_l2_5 = [get_augment_config('glitch', prob=0.02, std_fac=1.5, add_next=True),
                    get_augment_config('sine', freqs=[156, 220, 400], std_facs=[1.8, 2, 1.9])]
    fault_l2_6 = [get_augment_config('glitch', prob=0.05, std_fac=1.9, add_next=True),
                    get_augment_config('sine', freqs=[45, 300, 550, 700], std_facs=[1.6, 2.1, 1.3, 0.8]),]
    
    all_l2_faults = [fault_l2_1, fault_l2_2, fault_l2_3, fault_l2_4, fault_l2_5, fault_l2_6, fault_l2_1, fault_l2_2]

    # ctrl faults (l4 subtler)
    fault_l3_1 = [get_augment_config('glitch', prob=0.07, std_fac=0.7, add_next=True),
                    get_augment_config('sine', freqs=[30, 200, 500], std_facs=[0.7, 0.6, 0.6])]
    fault_l3_2 = [get_augment_config('glitch', prob=0.08, std_fac=0.77, add_next=True),
                    get_augment_config('sine', freqs=[56, 100], std_facs=[0.7, 0.9])]
    fault_l3_3 = [get_augment_config('glitch', prob=0.09, std_fac=0.6, add_next=True),
                    get_augment_config('sine', freqs=[150, 405, 556], std_facs=[0.9, 1, 0.77])]
    fault_l3_4 = [get_augment_config('glitch', prob=0.03, std_fac=0.6, add_next=True),
                    get_augment_config('sine', freqs=[100, 235, 375], std_facs=[0.7, 0.95, 0.7])]
    fault_l3_5 = [get_augment_config('glitch', prob=0.02, std_fac=0.9, add_next=True),
                    get_augment_config('sine', freqs=[156, 220, 400], std_facs=[0.6, 0.7, 0.9])]
    fault_l3_6 = [get_augment_config('glitch', prob=0.05, std_fac=0.7, add_next=True),
                    get_augment_config('sine', freqs=[45, 300, 550, 700], std_facs=[0.7, 0.77, 0.9, 0.5]),]
    
    all_l3_faults = [fault_l3_1, fault_l3_2, fault_l3_3, fault_l3_4, fault_l3_5, fault_l3_6, fault_l3_1, fault_l3_2]

    
    # ctrl faults (l4 subtler)
    fault_l4_1 = [get_augment_config('glitch', prob=0.07, std_fac=0.5, add_next=True),
                    get_augment_config('sine', freqs=[30, 200, 500], std_facs=[0.5, 0.4, 0.4])]
    fault_l4_2 = [get_augment_config('glitch', prob=0.08, std_fac=0.55, add_next=True),
                    get_augment_config('sine', freqs=[56, 100], std_facs=[0.5, 0.7])]
    fault_l4_3 = [get_augment_config('glitch', prob=0.09, std_fac=0.4, add_next=True),
                    get_augment_config('sine', freqs=[150, 405, 556], std_facs=[0.7, 0.8, 0.55])]
    fault_l4_4 = [get_augment_config('glitch', prob=0.03, std_fac=0.4, add_next=True),
                    get_augment_config('sine', freqs=[100, 235, 375], std_facs=[0.5, 0.75, 0.6])]
    fault_l4_5 = [get_augment_config('glitch', prob=0.02, std_fac=0.7, add_next=True),
                    get_augment_config('sine', freqs=[156, 220, 400], std_facs=[0.4, 0.5, 0.7])]
    fault_l4_6 = [get_augment_config('glitch', prob=0.05, std_fac=0.5, add_next=True),
                    get_augment_config('sine', freqs=[45, 300, 550, 700], std_facs=[0.5, 0.55, 0.7, 0.3]),]
    
    all_l4_faults = [fault_l4_1, fault_l4_2, fault_l4_3, fault_l4_4, fault_l4_5, fault_l4_6, fault_l4_1, fault_l4_2]

    if level == 2:
        return all_l2_faults
    if level == 3:
        return all_l3_faults
    elif level == 4:
        return all_l4_faults


def ws_ls1_pos_faults(level):
    fault_l2_1 = [get_augment_config('glitch', prob=0.07, std_fac=2, add_next=True),
                    get_augment_config('sine', freqs=[3, 2], std_facs=[1.8, 2])]
    fault_l2_2 = [get_augment_config('glitch', prob=0.08, std_fac=1.8, add_next=True),
                    get_augment_config('sine', freqs=[2, 4], std_facs=[1.8, 1.5])]
    fault_l2_3 = [get_augment_config('glitch', prob=0.09, std_fac=2.2, add_next=True),
                    get_augment_config('sine', freqs=[2, 4, 5], std_facs=[2, 1.8, 1.5])]
    fault_l2_4 = [get_augment_config('glitch', prob=0.03, std_fac=1.9, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 3], std_facs=[1.7, 1.9, 2])]
    fault_l2_5 = [get_augment_config('glitch', prob=0.02, std_fac=1.5, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 4], std_facs=[1.8, 2, 1.9])]
    fault_l2_6 = [get_augment_config('glitch', prob=0.05, std_fac=1.9, add_next=True),
                    get_augment_config('sine', freqs=[1, 3, 5, 7], std_facs=[1.6, 2.1, 1.8, 1.9]),]
    
    all_l2_faults = [fault_l2_1, fault_l2_2, fault_l2_3, fault_l2_4, fault_l2_5, fault_l2_6, fault_l2_1, fault_l2_2]

    # (l3 subtle)
    fault_l3_1 = [get_augment_config('glitch', prob=0.07, std_fac=1.2, add_next=True),
                    get_augment_config('sine', freqs=[3, 2], std_facs=[1.3, 1.4])]
    fault_l3_2 = [get_augment_config('glitch', prob=0.08, std_fac=1.5, add_next=True),
                    get_augment_config('sine', freqs=[2, 4], std_facs=[1.2, 1.3])]
    fault_l3_3 = [get_augment_config('glitch', prob=0.09, std_fac=1.35, add_next=True),
                    get_augment_config('sine', freqs=[2, 4, 5], std_facs=[1.2, 1.5, 1])]
    fault_l3_4 = [get_augment_config('glitch', prob=0.03, std_fac=1.25, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 3], std_facs=[1.3, 1.4, 1.25])]
    fault_l3_5 = [get_augment_config('glitch', prob=0.02, std_fac=1.2, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 4], std_facs=[1, 1.4, 1.1])]
    fault_l3_6 = [get_augment_config('glitch', prob=0.05, std_fac=1.3, add_next=True),
                    get_augment_config('sine', freqs=[1, 3, 5, 7], std_facs=[1.2, 1.1, 1, 1.3]),]
    
    all_l3_faults = [fault_l3_1, fault_l3_2, fault_l3_3, fault_l3_4, fault_l3_5, fault_l3_6, fault_l3_3, fault_l3_4]

    # (l4 subtle)
    fault_l4_1 = [get_augment_config('glitch', prob=0.07, std_fac=0.5, add_next=True),
                    get_augment_config('sine', freqs=[3, 2], std_facs=[0.65, 0.7])]
    fault_l4_2 = [get_augment_config('glitch', prob=0.08, std_fac=0.55, add_next=True),
                    get_augment_config('sine', freqs=[2, 4], std_facs=[0.8, 0.6])]
    fault_l4_3 = [get_augment_config('glitch', prob=0.09, std_fac=0.75, add_next=True),
                    get_augment_config('sine', freqs=[2, 4, 5], std_facs=[0.9, 0.8, 0.7])]
    fault_l4_4 = [get_augment_config('glitch', prob=0.03, std_fac=0.67, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 3], std_facs=[0.88, 0.65, 0.55])]
    fault_l4_5 = [get_augment_config('glitch', prob=0.02, std_fac=0.5, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 4], std_facs=[0.6, 0.7, 0.5])]
    fault_l4_6 = [get_augment_config('glitch', prob=0.05, std_fac=0.67, add_next=True),
                    get_augment_config('sine', freqs=[1, 3, 5, 7], std_facs=[0.7, 0.85, 0.6, 0.5]),]
    
    all_l4_faults = [fault_l4_1, fault_l4_2, fault_l4_3, fault_l4_4, fault_l4_5, fault_l4_6, fault_l4_4, fault_l4_5]

    # (l5 subtle)
    fault_l5_1 = [get_augment_config('glitch', prob=0.07, std_fac=0.2, add_next=True),
                    get_augment_config('sine', freqs=[3, 2], std_facs=[0.2, 0.3])]
    fault_l5_2 = [get_augment_config('glitch', prob=0.08, std_fac=0.2, add_next=True),
                    get_augment_config('sine', freqs=[2, 4], std_facs=[0.2, 0.2])]
    fault_l5_3 = [get_augment_config('glitch', prob=0.09, std_fac=0.3, add_next=True),
                    get_augment_config('sine', freqs=[2, 4, 5], std_facs=[0.4, 0.3, 0.25])]
    fault_l5_4 = [get_augment_config('glitch', prob=0.03, std_fac=0.23, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 3], std_facs=[0.45, 0.3, 0.2])]
    fault_l5_5 = [get_augment_config('glitch', prob=0.02, std_fac=0.2, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 4], std_facs=[0.2, 0.3, 0.2])]
    fault_l5_6 = [get_augment_config('glitch', prob=0.05, std_fac=0.3, add_next=True),
                    get_augment_config('sine', freqs=[1, 3, 5, 7], std_facs=[0.3, 0.25, 0.2, 0.2]),]
    
    all_l5_faults = [fault_l5_1, fault_l5_2, fault_l5_3, fault_l5_4, fault_l5_5, fault_l5_6, fault_l5_4, fault_l5_5]

    # (l5 subtle)
    fault_l6_1 = [#get_augment_config('glitch', prob=0.07, std_fac=0.2, add_next=True),
                    get_augment_config('sine', freqs=[3], std_facs=[0.2, 0.1])]
    fault_l6_2 = [#get_augment_config('glitch', prob=0.08, std_fac=0.2, add_next=True),
                    get_augment_config('sine', freqs=[4], std_facs=[0.1, 0.1])]
    fault_l6_3 = [get_augment_config('glitch', prob=0.09, std_fac=0.3, add_next=True),
                    get_augment_config('sine', freqs=[2, 4, 5], std_facs=[0.1, 0.3, 0.1])]
    fault_l6_4 = [#get_augment_config('glitch', prob=0.03, std_fac=0.23, add_next=True),
                    get_augment_config('sine', freqs=[1, 2, 3], std_facs=[0.1, 0.3, 0.2])]
    fault_l6_5 = [get_augment_config('glitch', prob=0.02, std_fac=0.2, add_next=True),
                    get_augment_config('sine', freqs=[1, 4], std_facs=[0.1, 0.2, 0.1])]
    fault_l6_6 = [#get_augment_config('glitch', prob=0.05, std_fac=0.3, add_next=True),
                    get_augment_config('sine', freqs=[1, 3, 5, 7], std_facs=[0.1, 0.2, 0.1, 0.1]),]
    
    all_l6_faults = [fault_l6_1, fault_l6_2, fault_l6_3, fault_l6_4, fault_l6_5, fault_l6_6, fault_l6_4, fault_l6_5]

    if level == 2:
        return all_l2_faults
    elif level == 3:
        return all_l3_faults
    elif level == 4:
        return all_l4_faults
    elif level == 5:
        return all_l5_faults
    elif level == 6:
        return all_l6_faults


def get_augment_config(augment_type, **kwargs):
    """
    Get the configuration for a specific augmentation type.

    Parameters
    ----------
    augment_type : str
        Type of augmentation to get configuration for.

    **kwargs : dict
        For all `augment_type`, the following parameters are available:
        - `OG` (Original data): No additional parameters
        - `gau` (Gaussian noise): **mean**, **snr_db**
        - `sine` (Sine wave): **freqs** (_list_), **std_facs** (_list_)
        - `glitch` (Random glitches): **prob**, **std_fac**

    Note
    ----
    std_factor is a multiplier that determines the strength of the augmentation (sine wave or glitch)
    relative to the standard deviation of the original signal.

    For `sine` and `glitch`, the amplitude is calculated as follows:
    - `sine` amplitude = signal std * std_factor * sqrt(2)
    - `glitch` amplitude = signal std * std_factor

    Examples:
    - std_factor = 1.0: The augmentation has the same standard deviation as the original signal.
    - std_factor < 1.0: The augmentation is weaker than the original signal.
    - std_factor > 1.0: The augmentation is stronger than the original signal.

    """
    config = {}
    config['type'] = augment_type

    if augment_type == 'gau':
        config['mean'] = kwargs.get('mean', 0.0)
        config['snr_db'] = kwargs.get('snr_db', 10)
    elif augment_type == 'sine':
        config['freqs'] = kwargs.get('freqs', [10.0])
        config['std_facs'] = kwargs.get('std_facs', [5.0])
    elif augment_type == 'glitch':
        config['prob'] = kwargs.get('prob', 0.01)
        config['std_fac'] = kwargs.get('std_fac', 5.0)

    config['add_next'] = kwargs.get('add_next', False)  # whether to add the next augmentation to the current one
    
    return config