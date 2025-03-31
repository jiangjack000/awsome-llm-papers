
from typing import List
import random
class Solution:
    def quick_sort(self, nums: List[int]):
        self.quick_sort_imp(nums, 0, len(nums)- 1)
        print(nums)
    def quick_sort_imp(self, nums: List[int], left:int, right:int):
        if left < right:
            pivot = self.partition(nums, left, right)
            self.quick_sort_imp(nums, left,pivot - 1)
            self.quick_sort_imp(nums, pivot + 1, right)
    def partition(self, nums:List[int], left:int, right:int):
        i, j, curr = left - 1, right + 1, random.choice(nums[left: right + 1])
        while i < j:
            i += 1
            while nums[i] < curr:
                i += 1
            j -= 1
            while nums[j] > curr:
                j -= 1
            if i < j:
                nums[i], nums[j] = nums[j], nums[i]
        return j
    def quick_select_imp(self, nums: List[int], left: int, right: int, k: int):
        if left == right:
            return nums[left]
        pivot = self.partition(nums, left, right)
        if k == pivot:
            return nums[k]
        elif k < pivot:
            return self.quick_select_imp(nums, left, pivot - 1, k)
        else:
            return self.quick_select_imp(nums, pivot + 1, right, k)
    
    def quick_select(self, nums: List[int], k: int):
        return self.quick_select_imp(nums, 0, len(nums) - 1, k)

if __name__ == '__main__':
    s = Solution()
    # s.quick_sort([3, 2, 4, 1, 5, 6, 7, 8, 9, 10])
    print(s.quick_select([3, 2, 4, 1, 5, 6, 7, 8, 9, 10], 7))
    array = [10, 20, 30, 40, 50]
print("Original array:", array)



