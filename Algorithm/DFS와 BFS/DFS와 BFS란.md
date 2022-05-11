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
