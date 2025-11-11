import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib import rcParams

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import rcParams

import numpy as np
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go

# 定义6个顶点
# z=1 时的三角形底面
v0 = [0, 0, 1]
v1 = [0, 1, 1]
v2 = [1, 1, 1]

# z=2 时的三角形顶面
v3 = [0, 0, 2]
v4 = [0, 2, 2]
v5 = [2, 2, 2]

vertices = np.array([v0, v1, v2, v3, v4, v5])

# 定义8个三角形面（组成这个楔形体）
# 底面 (z=1)
# 顶面 (z=2)
# 三个侧面
faces = [
    # 底面 (z=1)
    [0, 1, 2],

    # 顶面 (z=2)
    [3, 4, 5],

    # 侧面1: x=0 平面 (四边形，分成两个三角形)
    [0, 1, 4],
    [0, 4, 3],

    # 侧面2: x=y 平面 (四边形，分成两个三角形)
    [1, 2, 5],
    [1, 5, 4],

    # 侧面3: y=z 平面 (四边形，分成两个三角形)
    [0, 2, 5],
    [0, 5, 3]
]

# 创建网格数据
i = [face[0] for face in faces]
j = [face[1] for face in faces]
k = [face[2] for face in faces]

# 创建3D网格
fig = go.Figure(data=[
    go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i, j=j, k=k,
        opacity=0.7,
        color='cyan',
        flatshading=True,
        name='楔形区域'
    )
])

# 添加顶点
labels = [
    '(0,0,1)', '(0,1,1)', '(1,1,1)',
    '(0,0,2)', '(0,2,2)', '(2,2,2)'
]

fig.add_trace(go.Scatter3d(
    x=vertices[:, 0],
    y=vertices[:, 1],
    z=vertices[:, 2],
    mode='markers+text',
    marker=dict(size=8, color='red', line=dict(color='black', width=2)),
    text=labels,
    textposition='top center',
    textfont=dict(size=12, color='black'),
    name='顶点'
))

# 添加所有边
edges = [
    # 底面边
    [0, 1], [1, 2], [2, 0],
    # 顶面边
    [3, 4], [4, 5], [5, 3],
    # 竖直边
    [0, 3], [1, 4], [2, 5]
]

for edge in edges:
    fig.add_trace(go.Scatter3d(
        x=[vertices[edge[0], 0], vertices[edge[1], 0]],
        y=[vertices[edge[0], 1], vertices[edge[1], 1]],
        z=[vertices[edge[0], 2], vertices[edge[1], 2]],
        mode='lines',
        line=dict(color='darkblue', width=3),
        showlegend=False
    ))

# 设置布局
fig.update_layout(
    title=dict(
        text='空间区域: 0≤x≤y≤z, 1≤z≤2 (三棱柱/楔形体)<br><sub>鼠标拖动旋转 | 滚轮缩放</sub>',
        x=0.5,
        xanchor='center',
        font=dict(size=18)
    ),
    scene=dict(
        xaxis=dict(title='X', range=[-0.5, 2.5]),
        yaxis=dict(title='Y', range=[-0.5, 2.5]),
        zaxis=dict(title='Z', range=[0.5, 2.5]),
        aspectmode='cube',
        camera=dict(
            eye=dict(x=1.5, y=-1.5, z=1.2)
        )
    ),
    width=1200,
    height=900,
    showlegend=True
)

fig.show()