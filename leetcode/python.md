
``` python

def twos_complement_to_int(binary_str):
    # 确保字符串长度为32位
    assert len(binary_str) == 32, "字符串长度必须是32位"
    # 如果最高位是 '1'，这表明这是一个负数
    if binary_str[0] == '1':
        # 将其视为负数
        return -((1 << 32) - int(binary_str, 2))
    else:
        # 如果最高位是 '0'，那么这是一个正数
        return int(binary_str, 2)

# 示例
binary_str = '11111111111111111111111111111111'  # 这是32位全1 （即-1）
integer_value = twos_complement_to_int(binary_str)
print(integer_value)  # 输出 -1


# 字符串转换为整数
str_to_int = "123"
int_value = int(str_to_int)
print(f"字符串转换为整数: {int_value}")  # 输出 123

# 整数转换为字符串
int_to_str = int_value
str_value = str(int_to_str)
print(f"整数转换为字符串: {str_value}")  # 输出 '123'

# 字符串转换为浮点数
str_to_float = "123.45"
float_value = float(str_to_float)
print(f"字符串转换为浮点数: {float_value}")  # 输出 123.45

# 浮点数转换为字符串
float_to_str = float_value
str_value = str(float_to_str)
print(f"浮点数转换为字符串: {str_value}")  # 输出 '123.45'

# 整数转换为浮点数
int_to_float = int_value
float_value = float(int_to_float)
print(f"整数转换为浮点数: {float_value}")  # 输出 123.0

# 浮点数转换为整数
float_to_int = float_value
int_value = int(float_to_int)
print(f"浮点数转换为整数: {int_value}")  # 输出 123

# 字符串转换为布尔值
str_to_bool = "True"
bool_value = bool(str_to_bool)
print(f"字符串转换为布尔值: {bool_value}")  # 输出 True
# 注意：非空字符串在条件判断中默认为 True

# 整数转换为布尔值
int_to_bool = 0
bool_value = bool(int_to_bool)
print(f"整数转换为布尔值: {bool_value}")  # 输出 False
# 只有整数 0 转换为布尔值时得到 False，其他整数结果为 True

number = ord('a')
print(number)  # 输出 97

character = chr(97)
print(character)  # 输出 'a'

random.choice(array)

a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
a.sort()
print(a)  # 输出: [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]

a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
sorted_a = sorted(a)
print(sorted_a)  # 输出: [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
print(a)         # 原始列表 `a` 未被修改, 输出: [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]


class Solution:
    def quick_select(self, nums: List[int], k: int, i: int, j: int) -> int:
        if i == j:
            return nums[i]
        
        left, right = i, j
        pivot = nums[random.randint(i, j)]  # 随机选择一个基准值提高均匀性
        while left <= right:
            while nums[left] < pivot:
                left += 1
            while nums[right] > pivot:
                right -= 1
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

        if k <= right:
            return self.quick_select(nums, k, i, right)
        elif k >= left:
            return self.quick_select(nums, k, left, j)
        else:
            return nums[k]

    def findKthLargest(self, nums: List[int], k: int) -> int:
        size = len(nums)
        return self.quick_select(nums, size - k, 0, size - 1)


class Solution:
    def quick_sort(self, nums: List[int]):
        quick_sort_imp(nums, 0, len(nums)- 1)


import heapq
from collections import deque

# 数组（在 Python 中通常用列表表示）
array = [10, 20, 30, 40, 50]
print("Original array:", array)

# 添加元素
array.append(60)
print("After appending 60:", array)

# 访问元素
print("First element:", array)

# 修改元素
array = 100
print("After modifying first element:", array)

# 删除元素
array.pop()  # 删除最后一个元素
print("After popping last element:", array)

# 元组（不可变）
tuple_example = (1, 2, 3, 4, 5)
print("Tuple:", tuple_example)

# 访问元素
print("First element of tuple:", tuple_example)

# 尝试修改元组（会报错）
try:
    tuple_example = 10
except TypeError as e:
    print("Tuples are immutable, cannot modify elements:", e)

# 列表（可变）
list_example = [1, 2, 3, 4, 5]
print("Original list:", list_example)

# 添加元素
list_example.append(6)
print("After appending 6:", list_example)

# 插入元素
list_example.insert(2, 99)
print("After inserting 99 at index 2:", list_example)

# 删除元素
list_example.remove(3)  # 删除值为 3 的元素
print("After removing element 3:", list_example)

# 集合（无序且唯一）
set_example = {1, 2, 3, 4, 5}
print("Original set:", set_example)

# 添加元素
set_example.add(6)
print("After adding 6:", set_example)

# 删除元素
set_example.remove(2)  # 如果元素不存在，会抛出 KeyError
print("After removing 2:", set_example)

# 集合运算
set2 = {4, 5, 6, 7, 8}
print("Union:", set_example | set2)
print("Intersection:", set_example & set2)

# 字典（键值对）
dict_example = {'a': 1, 'b': 2, 'c': 3}
print("Original dictionary:", dict_example)

# 添加键值对
dict_example['d'] = 4
print("After adding key 'd':", dict_example)

# 修改值
dict_example['a'] = 100
print("After modifying key 'a':", dict_example)

# 删除键值对
del dict_example['b']
print("After deleting key 'b':", dict_example)

# 访问值
print("Value for key 'c':", dict_example['c'])

# 队列（使用 deque 模拟）
queue = deque(['a', 'b', 'c'])
print("Original queue:", queue)

# 入队
queue.append('d')
print("After appending 'd':", queue)

# 出队
queue.popleft()
print("After popping left:", queue)

# 优先队列（使用 heapq 模块）
priority_queue = []
heapq.heappush(priority_queue, (2, 'task2'))  # 优先级为 2
heapq.heappush(priority_queue, (1, 'task1'))  # 优先级为 1
heapq.heappush(priority_queue, (3, 'task3'))  # 优先级为 3

print("Priority queue:", priority_queue)

# 弹出优先级最高的元素
highest_priority = heapq.heappop(priority_queue)
print("Popped highest priority element:", highest_priority)

# 堆（与优先队列类似，使用 heapq）
heap = [5, 7, 9, 1, 3]
heapq.heapify(heap)  # 将列表转换为堆
print("Heap after heapify:", heap)

# 弹出堆顶元素
heap_top = heapq.heappop(heap)
print("Popped heap top element:", heap_top)


class Solution {
    // K从1开始，返回要找到的从小到大的第K个数
    public double findMedian(int[] nums1, int[] nums2, int k) {
        int m = nums1.length, n = nums2.length;
        int i = 0, j = 0;
        while(true) {
            // nums1用完了，直接访问nums2中的元素
            if(i == m) return nums2[k + j - 1];
            // nums2用完了，直接访问nums1中的元素
            if(j == n) return nums1[k + i - 1];
            // 两个数组都没用完，K = 1，比较2️两数组目前最前面的元素
            if(k == 1) return Math.min(nums1[i], nums2[j]);
            // 二分查找，此时的二分是基于K的二分，并非基于数组长度的二分
            // 1.为什么不是基于数组长度的二分？基于数组长度的二分找到的元素和K没有必然关系，需要再次判断
            int half = k / 2;
            // 防止溢出
            int mid1 = Math.min(i + half, m) - 1; 
            // 防止溢出
            int mid2 = Math.min(j + half, n) - 1;
            // 1.当 nums1[mid1] < nums2[mid2]，说明mid1所在位置一定不是第K个元素，并且它前面也不可能是
            // 原因：反证法，假设nums1[mid1] 是第K个元素，则nums1之前有k/2 - 1个元素，而nums2最多 k/2 - 1个元素
            // k / 2 - 1 + k /2 - 1 = k - 2，他前面最多有k -2个元素，而不是k - 1个元素，故不可能是
            // 2.当 nums1[mid1] = nums2[mid2]，和前面完全一样，他前面最多有k - 2个元素，而不是k - 1个元素，故不可能是
            // 3.当 nums1[mid1] = nums2[mid2]，说明mid2所在位置一定不是第K个元素，并且它前面也不可能是
            // 原因：和小于类似，假设nums2[mid2] 是第K个元素，则nums2之前有k/2 - 1个元素，而nums1最多 k/2 - 1个元素
            // k / 2 - 1 + k /2 - 1 = k - 2，他前面最多有k -2个元素，而不是k - 1个元素，故不可能是。
            // 将1，2两种合并在一起
            if(nums1[mid1] <= nums2[mid2]) {
                // 排除了【i，mid1】这个区间
                k -= (mid1 - i + 1);
                i = mid1 + 1;
            } else {
                // 排除了【j，mid2】这个区间
                k -= (mid2 - j + 1);
                j = mid2 + 1;
            }
            
        }
        
    }
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len = nums1.length + nums2.length;
        // 两数组长度之和如果是偶数，则找到第n / 2 和 n / 2 + 1 取平均数
        // 两数组长度之和如果是奇数，则找到第n / 2 + 1 
        if(len % 2 == 0) {
            return (findMedian(nums1, nums2, len / 2) 
             + findMedian(nums1, nums2, len / 2 + 1)) / 2;
        } else {
            return findMedian(nums1, nums2, len / 2 + 1); 
        }
    }
}

# 在 longcat.py 中修改导入语句，同一目录下
from .local_server import LocalServer

```