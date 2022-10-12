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



# 연습문제
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

loaded_model = load_model("c:\\projects\\model\\221012\\income_model.h5")
with open("c:\\projects\\model\\221012\\income_scaler.pickle", 'rb') as handle:
    loaded_scaler = pickle.load(handle)
with open("c:\\projects\\model\\221012\\income_onehot.pickle", 'rb') as handle:
    loaded_columns = pickle.load(handle)

user_input = []
user_question = {0: 'age', 1: 'workclass', 2: 'education', 3: 'educationNumber', 4: 'maritalStatus', 5: 'occupation', 6: 'relationship', 7: 'race', 8: 'gender', 9: 'hourPerWeek', 10: 'nativeCountry'}
for i in range(11):
    user_input.append(input(f"{user_question[i]}를 입력하세요: "))
#  ["50", "Self-emp-not-inc", "Bachelors", "13", "Married-civ-spouse",
#   "Exec-managerial", "Husband", "4", "Male", "13", "United-States"]

test_df = pd.DataFrame(user_input).T
test_df = test_df.rename(columns={0: 'age', 1: 'workclass', 2: 'education', 3: 'educationNumber', 4: 'maritalStatus', 5: 'occupation', 6: 'relationship', 7: 'race', 8: 'gender', 9: 'hourPerWeek', 10: 'nativeCountry'})
test_df.iloc[:,0] = test_df.iloc[:,0].apply(pd.to_numeric)
test_df.iloc[:,3] = test_df.iloc[:,3].apply(pd.to_numeric)
test_df.iloc[:,7] = test_df.iloc[:,7].apply(pd.to_numeric)
test_df.iloc[:,9] = test_df.iloc[:,9].apply(pd.to_numeric)

X_num = test_df.select_dtypes(exclude='object')
X_cat = test_df.select_dtypes(include='object')

X_scaled = loaded_scaler.transform(X_num)
X_hot = pd.get_dummies(X_cat)

for i in list(X_hot.columns):
    j = i
    tmp = i.replace('_', '_ ')
    X_hot = X_hot.rename(columns={j: tmp})

for columns in loaded_columns:
    if columns not in list(X_hot.columns):
        X_hot[columns] = 0

X_final = pd.concat([pd.DataFrame(X_scaled).reset_index(drop=True), X_hot.reset_index(drop=True)], axis=1)

predictions = loaded_model.predict(X_final)

if predictions >= 0.5:
    print(f"{float(predictions):.2f}% 확률로, above 50K")
else:
    print(f"{float(predictions):.2f}% 확률로, below 50K")
