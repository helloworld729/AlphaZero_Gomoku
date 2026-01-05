import pygame
import pygame.surfarray as surfarray


# 初始化
pygame.init()
pygame.display.set_caption('五子棋')


class Gomoku():
    def __init__(self, window_size=600, grid_size=8):
        # widow_size biao
        self.window_size = window_size
        self.grid_size = grid_size
        self.cell_size = self.window_size // self.grid_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))

        # 棋盘状态（0空，1黑，-1白）
        self.board = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.current_player = 1  # 黑棋先行
        self.running = True  # True表示没有结束
        self.data_queue = []

    # 绘制棋盘
    def draw_board(self):
        self.screen.fill((220, 179, 92))  # 木质背景
        for i in range(self.grid_size):
            pygame.draw.line(self.screen, (0, 0, 0), (self.cell_size//2, self.cell_size//2 + i*self.cell_size),
                             (self.window_size-self.cell_size//2, self.cell_size//2 + i*self.cell_size))
            pygame.draw.line(self.screen, (0, 0, 0), (self.cell_size//2 + i*self.cell_size, self.cell_size//2),
                             (self.cell_size//2 + i*self.cell_size, self.window_size-self.cell_size//2))
    
    # 赢棋检测
    def check_win(self, x, y):
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        reward = 0
        for dx, dy in directions:
            count = 1
            for step in [1, -1]:
                nx, ny = x + step*dx, y + step*dy
                while 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.board[nx][ny] == self.board[x][y]:
                    count += 1
                    nx += step*dx
                    ny += step*dy
            reward = max(reward, count)
            if count >= 5:
                print(reward)
                return True
        print(reward)
        return False
    
    def reset(self):
        self.board = [[0] * self.grid_size for _ in range(self.grid_size)]
        self.current_player = 1  # 黑棋先行
        self.running = True  # True表示没有结束

    # 走棋
    def step(self):
        self.update_screen()  # 更新界面
        state = surfarray.array3d(self.screen)
        # print(state.shape)
        reward = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos[0] // self.cell_size, event.pos[1] // self.cell_size
                print(x, y)
                if self.board[x][y] == 0:
                    self.board[x][y] = self.current_player
                    next_state = surfarray.array3d(self.screen)
                    self.data_queue.append({'state':state, 'action':(x, y), 'reward':reward, 'next_state':next_state})
                    if self.check_win(x, y):
                        reward = 1
                        print(f"Player {self.current_player} wins!")
                        print(len(self.data_queue))
                        self.running = False
                    self.current_player *= -1  # 切换玩家
                    self.update_screen()       # 更新界面

    # 更新屏幕
    def update_screen(self):
        # 绘制棋盘
        self.draw_board()

        # 绘制棋子
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.board[i][j] == 1:
                    pygame.draw.circle(self.screen, (0, 0, 0),
                                       (i * self.cell_size + self.cell_size // 2,
                                        j * self.cell_size + self.cell_size // 2), self.cell_size // 2 - 2)
                elif self.board[i][j] == -1:
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                       (i * self.cell_size + self.cell_size // 2,
                                        j * self.cell_size + self.cell_size // 2), self.cell_size // 2 - 2)


if __name__ == '__main__':
    gomoku = Gomoku()
    # 主循环
    while True:
        # 捕捉事件
        gomoku.step()

        # 更新屏幕
        pygame.display.flip()

        if not gomoku.running:
            gomoku.reset()



    # pygame.quit()
