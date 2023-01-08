# 通过图片像素分析法, 计算身高和部件比例
h_pix = 719
head_pix = 108
body_pix = 230
leg1_pix = 159
leg2_pix = 177
leg3_pix = 98
leg3_back_pix = 17
leg3_mid_pix = 58
leg3_front_pix = 25
# print("误差", leg3_front_pix + leg3_mid_pix + leg3_back_pix - leg3_pix)
leg_ankle_r_pix = 20


total_height = 1740
xpp = total_height/h_pix
print("单位毫米")
print('误差(毫米)', (leg3_front_pix + leg3_mid_pix + leg3_back_pix - leg3_pix) * xpp)
print('身高', h_pix*xpp)
print("头高", head_pix*xpp)
print("躯干长", body_pix*xpp)
print("大腿长", leg1_pix*xpp)
print("小腿长",leg2_pix*xpp)
print("脚长",leg3_pix*xpp)
print("脚后跟长",leg3_back_pix*xpp)
print("脚掌长",leg3_mid_pix*xpp)
print("脚趾长",leg3_front_pix*xpp)
print("脚踝直径",leg_ankle_r_pix*xpp)

"""
单位毫米
误差(毫米) 4.840055632823366
身高 1740.0
头高 261.36300417246173
躯干长 556.606397774687
大腿长 384.78442280945757
小腿长 428.3449235048679
脚长 237.16272600834492
脚后跟长 41.14047287899861
脚掌长 140.3616133518776
脚趾长 60.500695410292074
脚踝直径 48.400556328233655
"""