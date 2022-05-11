# DFS/BFS

## ⭐DFS (Depth First Search)

 **DFS(Depth First Search)**는 직역한 그대로 '깊이 우선 탐색' 알고리즘이다. 

그래프에서 깊은 부분을 우선적으로 탐색한다. 

DFS 구현 코드는 아래와 같다.

```python
# DFS 메소드 정의
def dfs(graph, v, visited):
    # 현재 노드를 방문 처리
    visited[v] = True
    print(v, end='')
    # 현재 노드와 연결된 다른 노드를 재귀적으로 방문
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)

# 각 노드가 연결된 정보를 리스트 자료형으로 표현 (2차원 리스트)

graph = [
    [],
    [2,3,8],
    [1,7],
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]

# 방문여부를 기록하기 위한 배열 선언
visited = [False]*9

# 정의된 DFS 함수 호출
dfs(graph,1,visited)
```

DFS의 핵심은 바로 **'재귀'와 '스택'**
이다. 타깃 노드를 기준으로 해당 노드와 연결된 다른 노드를 탐색하고, 또 그 노드와 연결된 다른 노드를 계속해서 탐색한다. 이때 dfs 함수를 재귀적으로 호출하는 방식으로 탐색을 한다. 만약 탐색한 노드들이 이미 방문된 노드들이라면, 다시 또 새로운 노드를 타깃으로 잡아 그 노드와 연결된 노드들을 모두 탐색한다. 스택은 선입후출이다. 즉 마지막에 들어온 노드가 가장 먼저 나가게 되는 것인데, 딱 DFS를 구현하기 알맞은 도구이다.

## ⭐BFS (Breadth First Search)

**BFS(Breadth First Search)**는 '넓이 우선 탐색' 알고리즘이다. 단어 그대로 넓이를 우선적으로 탐색하는 것이다. 기준이 되는 노드와 가까운 노드부터 탐색한다. DFS가 최대한 멀리 있는 노드까지 우선적으로 탐색한다면, BFS는 그 반대로, 최대한 인접한 노드 먼저 탐색한다. 이때 BFS의 핵심은 바로 '큐'를 이용하는 것이다. 큐는 선입선출이다. BFS는 선입선출을 활용하기에 알맞은 알고리즘이다. 그리고 이때 큐는 파이썬에서 제공하는 deque를 사용하는 것이 좋다고 한다. 탐색 시간이 O(N)이라서! 아래는 BFS 구현 코드이다.

```python
from collections import deque

# BFS 메소드 정의
def bfs(graph, start, visited):
    # 큐(Queue) 구현을 위해 deque 라이브러리 사용
    queue = deque([start])
    # 현재 노드를 방문 처리
    visited[start] = True
    # 큐가 빌 때까지 반복
    while queue:
        # 큐에서 원소 하나씩 뽑아 출력
        v = queue.popleft()
        print(v, end='')
        # 해당 원소와 연결된, 아직 방문하지 않은 원소들을 큐에 삽입
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True

# 각 노드가 연결된 정보를 리스트 자료형으로 표현 (2차원 리스트)
graph = [
    [],
    [2,3,8],
    [1,7],
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]

# 방문여부를 기록하기 위한 배열
visited = [False]*9

# 정의된 BFS 함수 호출
bfs(graph, 1, visited)
```

가장 먼저 들어온 노드를 popleft를 이용해 꺼내고, 꺼낸 노드와 인접한 노드를 큐에 넣는다. 그리고 그 다음으로 먼저 들어온 노드를 popleft로 꺼내서 이와 인접한 노드를 또 다시 큐에 넣는다. 이런식으로 큐가 빌 때까지 계속 반복한다. 얘는 재귀말고 while문을 사용한다.




# 🎯 문제) 백준 1012 유기농 배추

[1012번: 유기농 배추](https://www.acmicpc.net/problem/1012)

**[BFS 풀이]**

```python
from collections import deque

dx, dy = [-1, 1, 0, 0], [0, 0, -1, 1]

def bfs(x, y, _cnt):
    # 시작점
    q = deque()
    q.append((x, y))
    dist[x][y] = _cnt

    while q:
        x, y = q.popleft()

        for k in range(4):
            nx, ny = x+dx[k], y+dy[k]

            if 0<=nx<n and 0<=ny<m:
                if a[nx][ny] == 1 and dist[nx][ny] == -1:
                    q.append((nx, ny))
                    dist[nx][ny] = _cnt

t = int(input())
for _ in range(t):
    cnt = 0
    n, m, c = map(int, input().split())
    a = [[0] * m for _ in range(n)]
    dist = [[-1]*m for _ in range(n)]

    for _ in range(c):
        i, j = map(int, input().split())
        a[i][j] = 1

    # bfs 탐색
    for i in range(n):
        for j in range(m):
            if a[i][j] == 1 and dist[i][j] == -1:
                cnt += 1
                bfs(i, j, cnt)

    print(cnt)
```

**[BFS 풀이 2]**

```python
# BFS 풀이 2 (이게 더 간결한듯..?)
t = int(input())
dx = [1, -1, 0, 0]
dy = [0, 0, -1, 1]

def bfs(x, y):
    queue = [[x, y]]
    while queue:
        a, b = queue[0][0], queue[0][1]
        del queue[0]  # 이게 pop 과정인 것 같은디

        for i in range(4):
            q = a + dx[i]
            w = b + dy[i]
            if 0 <= q < n and 0 <= w < m and s[q][w] == 1: # 만약에 좌표가 주어진 범위 안이고, 배추가 있는 곳이면
                s[q][w] = 0 # 0으로 만들어주고
                queue.append([q, w])  # 큐에 좌표 추가

for i in range(t):
    m, n, k = map(int, input().split())
    s = [[0] * m for i in range(n)]
    cnt = 0
    for j in range(k): # 표 만들어줌
        a, b = map(int, input().split())
        s[b][a] = 1

    for q in range(n): # 하나씩 탐색 시작
        for w in range(m):
            if s[q][w] == 1: # 만약 배추가 있다면 (1이라면) bfs로 더 탐색해줌
                bfs(q, w)
                s[q][w] = 0 # 더이상 1이 없어서 bfs를 빠져나왔다면 0으로 만들어주고 
                cnt += 1 # cnt 횟수 1 증가
    print(cnt)
```

**[DFS 풀이]**

```python
import sys 
sys.setrecursionlimit(10000) 

def dfs(x, y): 
    dx = [1, -1, 0, 0] 
    dy = [0, 0, 1, -1] # 상,하,좌,우 확인 
    
    for i in range(4): 
        nx = x + dx[i] 
        ny = y + dy[i] 
        
        if (0 <= nx < N) and (0 <= ny < M): 
            if matrix[nx][ny] == 1: 
                matrix[nx][ny] = -1 
                dfs(nx, ny) 
                
T = int(input()) 
for _ in range(T): 
    M, N, K = map(int, input().split()) 
    matrix = [[0]*M for _ in range(N)] 
    cnt = 0 
    
    # 행렬 생성  
    for _ in range(K): 
        m, n = map(int, input().split()) 
        matrix[n][m] = 1 
        
        for i in range(N): # 행 (바깥 리스트) 
            for j in range(M): # 열 (내부 리스트) 
                if matrix[i][j] > 0: 
                    dfs(i, j) 
                    cnt += 1 
    print(cnt)
```
