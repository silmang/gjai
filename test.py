# class DOG:
#     def __init__(self, name, char, size, age, color):
#         self.name = name
#         self.char = char
#         self.size = size
#         self.age = age
#         self.color = color
#
#     def eat(self, food):
#         print(f"{self.name}이(가) {food}을(를) 먹는다.")
#
#
# class 말티즈(DOG):
#     def play(self):
#         print(f"{self.name}은(는) 놀이를 한다.")
#
#
# class 차우차우(DOG):
#     def attack(self):
#         print(f"{self.name}은(는) 공격을 한다.")
#
#
# class 마스티프(DOG):
#     def hunt(self):
#         print(f"{self.name}은(는) 사냥을 한다.")
#
#
# 개1 = 말티즈("말티즈", "온순", "크다", 10, "검정")
# 개1.play()
# 개1.eat('고기')
#
# 개2 = 차우차우("차우차우", "난폭", "작다", 3, "파랑")
# 개2.attack()
#
# 개3 = 마스티프("마스티프", "소심", "평범", 5, "노랑")
# 개3.hunt()

# # class Car:
# #     def __init__(self, name):
# #         self.name = name
# #
# #     def __call__(self):
# #         # print(self.name)
# #         return self.name
# #
# #     def make_noise(self):
# #         print("삐삐")
# #
# #
# # car1 = Car("자동차1")
# #
# # car_name = car1()
# # print(f"car1의 이름은 {car_name}")
# #
# # car1.make_noise()
#
# class Car:
#     def __init__(self, sound):
#         self.sound = sound
#         print(self.sound)
#
#     # def make_noise(self):
#     #     print("삐삐")
#
#
# class Suv(Car):
#     def __init__(self, sound):
#         self.sound = sound
#
#     def make_noise(self):
#         # super().make_noise()
#         # print("빵빵")
#         # super().make_noise()
#         print(self.sound)
#         super().__init__("삐삐")
#
#     def open_hatchback(self):
#         print("뒷문을 열었습니다.")
#
# suv = Suv("빵빵")
# suv.make_noise()
# # suv.open_hatchback()

# import os
# os.chdir(r"c:\drivers")
# print(os.getcwd())

# path = r"c:\pytest_basic"
# with open(path+"\\myfile2.txt", "w", encoding='ANSI') as f:
#     text = "이것은 첫 번째 문장입니다.\n그리고 이것은 두 번째 문장이고요.\n마지막 세 번째 문장이랍니다."
#     f.write(text)

# f = open(path+"\\myfile2.txt", "r")
# text = f.read()
# print(text)
# f.close()



# a, b ,c = list(map(int, input().split()))
# b_list = bin(b)[2:]
# count = 0
# two_list = []
# for num in b_list:
#     count += 1
#     if num == '1':
#         temp = 2**(len(b_list) - count)
#         two_list.append(temp)
# b_list = list(b_list)
# b_list.reverse()
# two_list.reverse()
# modulo_list = []
# cnt = 0
# def modulo(a, b, c, cnt):
#     if cnt == 0:
#         temp = a % c
#         modulo_list.append(temp)
#     else:
#         temp = modulo_list[cnt - 1]
#         modulo_list.append((temp**2)%c)
#     cnt += 1
#     if cnt == b:
#         return modulo_list
#     modulo(a, b, c, cnt)
# modulo(a, b, c, cnt)
# temp = 1
# result = 0
# for num, modulo in zip(b_list, modulo_list):
#     if num == '1':
#         temp *= modulo
#         result = temp
# print(result%c)

import find_words.find_keywords
# find_words.
help(find_words.find_keywords)

path =  "c:\\pytest\\"
# find_words_from_file.find(path+'금융규제운영규정.txt', path+'220915_연습문제6.csv', ['금융','법률'])
p = find_words.find_keywords.find_kewords_in_txt(path+'금융규제운영규정.txt', ['규제'])
print(p)