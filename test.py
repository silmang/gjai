# # class DOG:
# #     def __init__(self, name, char, size, age, color):
# #         self.name = name
# #         self.char = char
# #         self.size = size
# #         self.age = age
# #         self.color = color
# #
# #     def eat(self, food):
# #         print(f"{self.name}이(가) {food}을(를) 먹는다.")
# #
# #
# # class 말티즈(DOG):
# #     def play(self):
# #         print(f"{self.name}은(는) 놀이를 한다.")
# #
# #
# # class 차우차우(DOG):
# #     def attack(self):
# #         print(f"{self.name}은(는) 공격을 한다.")
# #
# #
# # class 마스티프(DOG):
# #     def hunt(self):
# #         print(f"{self.name}은(는) 사냥을 한다.")
# #
# #
# # 개1 = 말티즈("말티즈", "온순", "크다", 10, "검정")
# # 개1.play()
# # 개1.eat('고기')
# #
# # 개2 = 차우차우("차우차우", "난폭", "작다", 3, "파랑")
# # 개2.attack()
# #
# # 개3 = 마스티프("마스티프", "소심", "평범", 5, "노랑")
# # 개3.hunt()

# # # class Car:
# # #     def __init__(self, name):
# # #         self.name = name
# # #
# # #     def __call__(self):
# # #         # print(self.name)
# # #         return self.name
# # #
# # #     def make_noise(self):
# # #         print("삐삐")
# # #
# # #
# # # car1 = Car("자동차1")
# # #
# # # car_name = car1()
# # # print(f"car1의 이름은 {car_name}")
# # #
# # # car1.make_noise()
# #
# # class Car:
# #     def __init__(self, sound):
# #         self.sound = sound
# #         print(self.sound)
# #
# #     # def make_noise(self):
# #     #     print("삐삐")
# #
# #
# # class Suv(Car):
# #     def __init__(self, sound):
# #         self.sound = sound
# #
# #     def make_noise(self):
# #         # super().make_noise()
# #         # print("빵빵")
# #         # super().make_noise()
# #         print(self.sound)
# #         super().__init__("삐삐")
# #
# #     def open_hatchback(self):
# #         print("뒷문을 열었습니다.")
# #
# # suv = Suv("빵빵")
# # suv.make_noise()
# # # suv.open_hatchback()

# # import os
# # os.chdir(r"c:\drivers")
# # print(os.getcwd())

# # path = r"c:\pytest_basic"
# # with open(path+"\\myfile2.txt", "w", encoding='ANSI') as f:
# #     text = "이것은 첫 번째 문장입니다.\n그리고 이것은 두 번째 문장이고요.\n마지막 세 번째 문장이랍니다."
# #     f.write(text)

# # f = open(path+"\\myfile2.txt", "r")
# # text = f.read()
# # print(text)
# # f.close()



# # a, b ,c = list(map(int, input().split()))
# # b_list = bin(b)[2:]
# # count = 0
# # two_list = []
# # for num in b_list:
# #     count += 1
# #     if num == '1':
# #         temp = 2**(len(b_list) - count)
# #         two_list.append(temp)
# # b_list = list(b_list)
# # b_list.reverse()
# # two_list.reverse()
# # modulo_list = []
# # cnt = 0
# # def modulo(a, b, c, cnt):
# #     if cnt == 0:
# #         temp = a % c
# #         modulo_list.append(temp)
# #     else:
# #         temp = modulo_list[cnt - 1]
# #         modulo_list.append((temp**2)%c)
# #     cnt += 1
# #     if cnt == b:
# #         return modulo_list
# #     modulo(a, b, c, cnt)
# # modulo(a, b, c, cnt)
# # temp = 1
# # result = 0
# # for num, modulo in zip(b_list, modulo_list):
# #     if num == '1':
# #         temp *= modulo
# #         result = temp
# # print(result%c)

# # import find_words.find_keywords
# # # find_words.
# # help(find_words.find_keywords)

# # path =  "c:\\pytest\\"
# # # find_words_from_file.find(path+'금융규제운영규정.txt', path+'220915_연습문제6.csv', ['금융','법률'])
# # p = find_words.find_keywords.find_kewords_in_txt(path+'금융규제운영규정.txt', ['규제'])
# # print(p)
# import pickle
# from sklearn.utils import all_estimators
# all = all_estimators(type_filter='regressor')
# for i in all:
#     print(i)

# # with open('regressor.pickle', 'wb') as f:
# #     pickle.dump(all, f, pickle.HIGHEST_PROTOCOL)



# # 연습문제
# import pickle
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import load_model

# loaded_model = load_model("c:\\projects\\model\\221012\\income_model.h5")
# with open("c:\\projects\\model\\221012\\income_scaler.pickle", 'rb') as handle:
#     loaded_scaler = pickle.load(handle)
# with open("c:\\projects\\model\\221012\\income_onehot.pickle", 'rb') as handle:
#     loaded_columns = pickle.load(handle)

# user_input = []
# user_question = {0: 'age', 1: 'workclass', 2: 'education', 3: 'educationNumber', 4: 'maritalStatus', 5: 'occupation', 6: 'relationship', 7: 'race', 8: 'gender', 9: 'hourPerWeek', 10: 'nativeCountry'}
# for i in range(11):
#     user_input.append(input(f"{user_question[i]}를 입력하세요: "))
# #  ["50", "Self-emp-not-inc", "Bachelors", "13", "Married-civ-spouse",
# #   "Exec-managerial", "Husband", "4", "Male", "13", "United-States"]

# test_df = pd.DataFrame(user_input).T
# test_df = test_df.rename(columns={0: 'age', 1: 'workclass', 2: 'education', 3: 'educationNumber', 4: 'maritalStatus', 5: 'occupation', 6: 'relationship', 7: 'race', 8: 'gender', 9: 'hourPerWeek', 10: 'nativeCountry'})
# test_df.iloc[:,0] = test_df.iloc[:,0].apply(pd.to_numeric)
# test_df.iloc[:,3] = test_df.iloc[:,3].apply(pd.to_numeric)
# test_df.iloc[:,7] = test_df.iloc[:,7].apply(pd.to_numeric)
# test_df.iloc[:,9] = test_df.iloc[:,9].apply(pd.to_numeric)

# X_num = test_df.select_dtypes(exclude='object')
# X_cat = test_df.select_dtypes(include='object')

# X_scaled = loaded_scaler.transform(X_num)
# X_hot = pd.get_dummies(X_cat)

# for i in list(X_hot.columns):
#     j = i
#     tmp = i.replace('_', '_ ')
#     X_hot = X_hot.rename(columns={j: tmp})

# for columns in loaded_columns:
#     if columns not in list(X_hot.columns):
#         X_hot[columns] = 0

# X_final = pd.concat([pd.DataFrame(X_scaled).reset_index(drop=True), X_hot.reset_index(drop=True)], axis=1)

# predictions = loaded_model.predict(X_final)

# if predictions >= 0.5:
#     print(f"{float(predictions):.2f}% 확률로, above 50K")
# else:
#     print(f"{float(predictions):.2f}% 확률로, below 50K")


# 0 6 8 0 0 0 9 3 0
# 0 4 2 0 0 0 6 0 0
# 1 9 0 0 8 0 0 4 0
# 0 8 5 2 0 1 0 0 7
# 7 0 0 8 9 0 0 0 0
# 2 0 9 0 0 7 5 0 3
# 0 2 0 1 0 0 0 5 0
# 8 5 0 0 4 0 7 6 0
# 4 7 3 0 5 2 0 0 9

# 5 6 8 7 2 4 9 3 1 
# 3 4 2 5 1 9 6 7 8 
# 1 9 7 3 8 6 2 4 5 
# 6 8 5 2 3 1 4 9 7 
# 7 3 4 8 9 5 1 2 6 
# 2 1 9 4 6 7 5 8 3 
# 9 2 6 1 7 8 3 5 4 
# 8 5 1 9 4 3 7 6 2 
# 4 7 3 6 5 2 8 1 9 

# 스도쿠 마스터
# 퍼즐 게임을 좋아하는 체셔는 요즘 스도쿠에 푹 빠져있습니다. 스도쿠는 숫자퍼즐게임으로 다음과 같은 규칙을 가지고 있습니다.

# 스도쿠의 규칙

# 각각의 가로줄과 세로줄에 숫자 1~9가 중복 없이 하나씩 들어간다.
# 3X3 모양의 네모난 박스 안에는 1~9가 중복 없이 하나씩 들어간다.
# 체셔는 재밌는 스도쿠를 여러 친구들과 같이 즐기고 싶어서 문제와 답지를 같이 건네주려고 합니다. 하지만 체셔는 답지를 가지고 있지 않아 모든 문제의 답을 찾는 시간이 너무 아깝게 느껴졌습니다. 이런 체셔를 위해 여러분이 스도쿠의 답을 출력해주는 프로그램을 만들어주세요.

# 입력
# 9개의 숫자가 공백을 기준으로 나뉘어 9줄로 제공됩니다.
# 현재 스도쿠의 비어있는 칸은 0으로 표시됩니다.
# 정답을 찾을 수 없는 입력값은 주어지지 않습니다.
# 정답이 여러 개일 수 있는 입력은 주어지지 않습니다.
# 출력
# 모든 빈 칸이 채워진 스도쿠의 정답을 입력값과 동일한 형태인 9개의 숫자를 공백을 기준으로 9줄로 출력하세요.



# def check_x(x,y,n):
#     for i in range(9):
#         if n==box[x][i]:
#             return False
#     return True

# def check_y(x,y,n):
#     for i in range(9):
#         if n==box[i][y]:
#             return False
#     return True

# def check_sector(x,y,n):
#     x=x//3
#     y=y//3
#     for i in range(x*3,x*3+3):
#         for j in range(y*3,y*3+3):
#             if n==box[i][j]:
#                 return False
#     return True

# def track(n):
#     if n==len(blank):
#         for _ in range(9):
#             print(*box[_])
#         exit(0)
#     x,y=blank[n][0],blank[n][1]
#     for i in range(1,10):
#         if check_x(x,y,i) and check_y(x,y,i) and check_sector(x,y,i):
#             box[x][y] = i
#             track(n + 1)
#             box[x][y] = 0

# box=[]
# blank=[]
# for i in range(9):
#     box.append(list(map(int, input().split())))
#     for j in range(9):
#         if box[i][j]==0:
#             blank.append([i,j])
# track(0)



# import itertools

# num = list(input()) # 숫자 입력받기
# arr = list(itertools.permutations(num, len(num))) # 파이썬 라이브러리를 활용해 순열 구하기
# arr.sort(reverse=True) # 오름차순 정렬

# result = [] # 문자열을 숫자열로 합쳐서 저장할 배열 생성
# for a in arr: # arr에서 값을 가져오기
#   result.append("".join(a)) # 튜플을 문자열로 합침

# idx = result.index("".join(num)) # 숫자가 있는 인덱스 번호를 찾기
# print(result[idx-1] if idx != 0 else 0) # 제일 큰 수가 아니라면 현재보다 바로 큰 값을 출력하고 0을 출력한다.

import itertools
num_list = list(map(int, input().split()))
p=list(map(''.join, map(str,list(itertools.permutations(num_list,4)))))
hour=[]
for i in p:
    h=int(i[1]+i[4])
    if h<=23:
        hour.append(h)
for j in sorted(set(hour),reverse=True):
    temp_list = num_list.copy()
    if len(str(j))==1:
        temp_list.pop(temp_list.index(0))
        temp_list.pop(temp_list.index(j))
    else:
        temp_list.pop(temp_list.index(int(str(j)[0])))
        temp_list.pop(temp_list.index(int(str(j)[1])))
    tmp_min = []
    if temp_list[0]*10+temp_list[1] <= 60:
        tmp_min.append(temp_list[0]*10+temp_list[1])
    if temp_list[0]+temp_list[1]*10 <= 60:
        tmp_min.append(temp_list[0]+temp_list[1]*10)
    if len(tmp_min)==0:
        continue
    min = max(tmp_min)
    print(f"{j:02d}:{min:02d}")
    break
else:
    print(-1)