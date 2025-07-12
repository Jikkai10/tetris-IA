import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
colors = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
]


class Figure:
    x = 0
    y = 0

    figures = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],
        [[4, 5, 9, 10], [2, 6, 5, 9]],
        [[6, 7, 9, 10], [1, 5, 6, 10]],
        [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        [[1, 2, 5, 6]],
    ]

    def __init__(self, x, y,p,r=0):
        self.x = x
        self.y = y
        self.type = p
        self.color = random.randint(1, len(colors) - 1)
        self.rotation = r

    def image(self,type=None, rotation=None):
        if type is None:
            type = self.type
        if rotation is None:
            rotation = self.rotation
        return self.figures[type][rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])


class Tetris:
    def __init__(self, height, width):
        self.level = 2
        self.x = 100
        self.y = 60
        self.zoom = 20
        self.figure = None
        self.count = 0
        self.height = height
        self.width = width
        self.field = torch.zeros((height, width)).to(device)
        self.last_field = torch.zeros((height, width)).to(device)
        self.score = 0
        self.reward = 0
        self.state = "start"
        self.next_figure = random.randint(0, 6)
        

    def new_figure(self):
        p = self.next_figure
        self.next_figure = random.randint(0, 6) 
        self.figure = Figure(3, 0, p)

    def intersects(self,figure,field,height,width):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in figure.image():
                    if i + figure.y > height - 1 or \
                            j + figure.x > width - 1 or \
                            j + figure.x < 0 or \
                            field[i + figure.y][j + figure.x] > 0:
                        intersection = True
        return intersection
    
    
    def break_lines(self):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        self.reward += (lines+1) ** 2 

        self.score += lines ** 2
        
        

    def go_space(self):
        y = self.figure.y
        x = self.figure.x
        rotation = self.figure.rotation
        figure_type = self.figure.type
        
        
        field = self.field.clone()
        last_field = self.last_field.clone()
        
        # cria uma cópia da peça
        figure = Figure(x, y, figure_type, rotation)
       
        
        count = 0
        while not self.intersects(figure,field,self.height,self.width):
            figure.y += 1
            
            count += 1
        figure.y -= 1
        
        count -= 1
        
        shadow = [figure.y,figure.x]
        
        
        return [0, shadow, 0]
        

    def go_down(self):
        self.figure.y += 1
        self.count += 1
        
        if self.intersects(self.figure,self.field,self.height,self.width):
            self.reward = 0
            self.figure.y -= 1
            
            self.freeze()
            self.reward += self.calculate_reward(self.field,self.last_field) 
            new = self.go_space()
            self.last_field = self.field.clone()
            return [self.reward, new[1], 1]
        new = self.go_space()
        
        return new
    
    def calculate_reward(self, new_table, last_table):
        
        new_table = (new_table > 0).int()
        last_table = (last_table > 0).int()
        height, width = new_table.shape
        max_height = 0
        last_max_height = 0
        
            
            
        column_heights = height - torch.argmax(new_table[:, :], axis=0)
        
        
        last_column_heights = height - torch.argmax(last_table[:, :], axis=0)
        
        for i in range(width):
            if torch.sum(new_table[:,i], axis=0) == 0:
                column_heights[i] = 0
            if torch.sum(last_table[:,i], axis=0) == 0:
                last_column_heights[i] = 0
        max_height = torch.max(column_heights)
        last_max_height = torch.max(last_column_heights)
            
        #
        holes = 0
        last_holes = 0
        line = 0
        last_line = 0
        if max_height > 0:
            for row in range(20-max_height,20):
                for col in range(width):
                    if new_table[row,col] == 0 and torch.any(new_table[:row, col] > 0):
                        
                        holes += 1
                    
              
              
        
        if last_max_height > 0:
            for row in range(20-last_max_height,20):
                for col in range(width):
                    if last_table[row,col] == 0 and torch.any(last_table[:row, col] > 0):
                        last_holes += 1
                
        
       
        bumpiness = torch.sum(torch.abs(torch.diff(column_heights)))
        last_bumpiness = torch.sum(torch.abs(torch.diff(last_column_heights)))
    
        
        const_holes = torch.tensor((last_holes - holes) / 5.0)
        if holes - last_holes > 0 and holes - last_holes <= 2:
            const_holes = torch.tensor(-0.5)
        
        const_bump = (last_bumpiness-bumpiness)/8

        const_holes = torch.clip(const_holes, -1, 1)
        
        const_bump = torch.clip(const_bump, -1, 1)


        reward = 0.8*const_holes + 0.2*const_bump 

        return float(reward)

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.break_lines()
        self.new_figure()
        if self.intersects(self.figure,self.field,self.height,self.width):
            self.reward = -5
            self.state = "gameover"
            
    
    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects(self.figure,self.field,self.height,self.width):
            self.figure.x = old_x
            return 1
        return 0

    def rotate(self):
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects(self.figure,self.field,self.height,self.width):
            self.figure.rotation = old_rotation
            return 1
        return 0
            
GAMMA = 0.99
EPSILON = 1
EPSILON_DECAY = 100000

MIN_EPSILON = 0.01
LR = 5e-6
BATCH_SIZE = 64
MEMORY_SIZE = 200000
TARGET_UPDATE = 5000




from SumTree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.eps = 1e-5
        self.max_priority = 1.0

    def push(self, transition):
        p = self.max_priority
        self.tree.add(p, transition)

    def sample(self, batch_size, beta):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            s = np.random.uniform(i * segment, (i + 1) * segment)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()

        return batch, idxs, weights

    def update_priorities(self, idxs, td_errors):
        for idx, error in zip(idxs, td_errors):
            p = (np.abs(error) + self.eps) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.size
    


def train_step(dqn, target, optimizer, buffer, frame_idx, batch_size, gamma=GAMMA, beta_start=0.4, beta_frames=1000000):
    if len(buffer) < batch_size:
        return

    beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
    transitions, idxs, weights = buffer.sample(batch_size, beta)

    states, actions, rewards, next_states, dones = zip(*transitions)

    states = torch.stack(states).to(device)
    #extras = torch.stack(extras).to(device)
    actions = torch.tensor(actions).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards).unsqueeze(1).to(device)
    next_states = torch.stack(next_states).to(device)
    #new_extras = torch.stack(new_extras).to(device)
    dones = torch.tensor(dones).unsqueeze(1).to(device)
    weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

    q_values = dqn(states).gather(1, actions)
    with torch.no_grad():
        next_actions = dqn(next_states).argmax(1, keepdim=True)
        next_q = target(next_states).gather(1, next_actions)
        target_q = rewards + gamma * next_q * (~dones)

    td_errors = target_q - q_values
    loss = (weights * td_errors.pow(2)).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

    buffer.update_priorities(idxs, td_errors.detach().squeeze().cpu().numpy())



# --- Rede neural para estimar Q-values ---
class DQN(nn.Module):
    def __init__(self, extra_dim, action_dim):
        super(DQN, self).__init__()
        # Parte compartilhada da CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            
            nn.Flatten()
        )

        
        

      

        # Cálculo do tamanho de saída
        dummy_input = torch.zeros(1, 4, 20, 10)
        flat = self.cnn(dummy_input).shape[1]
        

        total_features =  flat #+ 64

        # Camada combinada
        self.combined = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU()
          
        )

        # Dueling DQN
        self.value_stream = nn.Sequential(
        
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            
         
            nn.Linear(256, action_dim)
        )

    def forward(self, image):
        
        cnn = self.cnn(image)
        

      
        x = self.combined(cnn)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        return value + advantage - advantage.mean(dim=1, keepdim=True)

class train:
    def __init__(self, game, dqn, mode=0):
        self.game = game
        self.episodes = 10000
        self.done = False
        self.epsilon = EPSILON
        state_dim = 7 + 7 + 4
        action_dim = 4
        self.dqn = dqn
        self.target = DQN(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.dqn.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=LR)
        
        self.buffer = PrioritizedReplayBuffer(capacity=MEMORY_SIZE, alpha=0.6)
        self.sum_reward = 0
        self.counter = 0
        self.count = 0
        self.down_count = 0
        self.figure_onehot = torch.eye(7, device=device)
        self.rotation_onehot = torch.eye(4, device=device)
        self.mode = mode

    def init_game(self):
        # reinicia tudo
        self.game = Tetris(20, 10)
        self.game.new_figure()
        self.done = False
        

        if self.count % 100 == 0:
            print(f"Episode {self.count}/{self.episodes} (counter={self.counter})")
            print(f"reward: {self.sum_reward} ")
            self.sum_reward = 0
        if self.count % 500 == 0:
            torch.save(self.dqn.state_dict(), f"tetris_dqn_{self.count}.pth")
        if self.count % 10 == 0:
            torch.save(self.dqn.state_dict(), f"tetris_dqn_A.pth")

        field = (self.game.field > 0).float()
        piece = torch.zeros_like(field)
        shadow = torch.zeros_like(field)
        next = torch.zeros_like(field)
        new_shadow = self.game.go_space()[1]
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in self.game.figure.image():
                    piece[i + self.game.figure.y][j + self.game.figure.x] = 1
                    shadow[i + new_shadow[0]][j + new_shadow[1]] = 1
                    
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in self.game.figure.image(self.game.next_figure, 0):
                    next[i + 0][j + 3] = 1

        self.state = torch.stack([field, piece, shadow, next]).to(device)

        
        self.count += 1

    def make_action(self, action):
        reward = 0
        game = self.game

        # movimento
        if action == 0:
            if game.rotate():
                self.down_count -= 0.001
            
        elif action == 1:
            if game.go_side(-1):
                self.down_count -= 0.001
            
        elif action == 2:
            if game.go_side(1):
                self.down_count -= 0.001
            
        new = game.go_down()
        self.down_count += 0.01
        reward += new[0] + self.down_count
        if new[2] == 1:
            self.down_count = 0
       
        field = (game.field > 0).float()
        piece = torch.zeros_like(field)
        shadow = torch.zeros_like(field)
        next = torch.zeros_like(field)
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in game.figure.image():
                    piece[i + game.figure.y][j + game.figure.x] = 1
                    shadow[i + new[1][0]][j + new[1][1]] = 1
                    
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in game.figure.image(game.next_figure, 0):
                    next[i + 0][j + 3] = 1

        new_state = torch.stack([field, piece, shadow, next]).to(device)

        

        done = game.state == "gameover"

        self.buffer.push((self.state, action, reward, new_state, done))

        self.state = new_state
       
        self.done = done

    def step(self):
       
        if (np.random.uniform() < self.epsilon or self.counter < 10000) and self.mode == 0:
            action = np.random.choice(range(4))
        else:
            with torch.no_grad():
                q = self.dqn(self.state.unsqueeze(0))
                action = q.argmax().item()

        self.make_action(action)

        self.counter += 1
        if self.counter > 10000:
            train_step(self.dqn, self.target, self.optimizer, self.buffer, self.counter, BATCH_SIZE)

        if self.counter < 500000:
            self.epsilon = max(MIN_EPSILON, self.epsilon * np.exp(-1.0 / EPSILON_DECAY))
            
        else:
            self.epsilon = 0.001

        if self.counter % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.dqn.state_dict())
            

    def train_loop(self):
        while self.count < self.episodes:
            self.init_game()
            while not self.done:
                self.step()
            self.sum_reward += self.game.score


def Q_trainer(game, dqn, mode=0):
    trainer = train(game, dqn, mode)
    trainer.train_loop()
    torch.save(trainer.dqn.state_dict(), "tetris_dqn_final.pth")



def Q_player(is_training=False, dqn_path=None):
    #torch.random.manual_seed(1) 
    game = Tetris(20, 10)
    
    if is_training:
        dqn = DQN(7+7+4, 4).to(device)
        mode = 0
        if dqn_path:
            dqn.load_state_dict(torch.load(dqn_path))
            mode = 1
        Q_trainer(game,dqn,mode)
        
        return
    
    dqn = DQN(7+7+4, 4).to(device)
    if dqn_path:
        dqn.load_state_dict(torch.load(dqn_path))
        dqn.eval()
    counter = 0
    pygame.init()

    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)

    size = (400, 500)
    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("Tetris")

    # Loop until the user clicks the close button.
    done = False
    clock = pygame.time.Clock()
    fps = 60
    action = 0
    
    while True:
        if game.figure is None:
            print("New figure")
            game.new_figure()
            #new  = game.go_space()[1]
        counter += 1
        

        if counter % (1) == 0 or action == 3:
            if game.state == "start":
                new = game.go_down()[1]
            counter = 0
        else:
            new  = game.go_space()[1]
        field = (game.field > 0).float()  
        piece = torch.zeros_like(field).to(device)
        shadow = torch.zeros_like(field).to(device)
        next = torch.zeros_like(field)
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in game.figure.image():
                    piece[i + game.figure.y][j + game.figure.x] = 1
                    shadow[i + new[0]][j + new[1]] = 1
                    
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in game.figure.image(game.next_figure, 0):
                    next[i + 0][j + 3] = 1



            
            
            
        
        state = torch.stack([field, piece, shadow,next], axis=0 ).to(device)
        

        
        with torch.no_grad():
            q_values = dqn(state.unsqueeze(0))
            print(f"Q-values: {q_values}")
            action = torch.argmax(q_values).item()


        if action == 0:
            game.rotate()
        elif action == 1:
            game.go_side(-1)
        elif action == 2:
            game.go_side(1)
            
        if game.state == "gameover":
            print("Game Over")
            game = Tetris(20, 10)
        
        screen.fill(WHITE)

        for i in range(game.height):
            for j in range(game.width):
                pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom], 1)
                if game.field[i][j] > 0:
                    #print(game.field[i][j].item())
                    pygame.draw.rect(screen, colors[int(game.field[i][j].item())],
                                    [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 2, game.zoom - 1])
         
        if game.figure is not None:
            for i in range(4):
                for j in range(4):
                    p = i * 4 + j
                    if p in game.figure.image():
                        
                        
                        pygame.draw.rect(screen, colors[game.figure.color],
                                        [game.x + game.zoom * (j + game.figure.x) + 1,
                                        game.y + game.zoom * (i + game.figure.y) + 1,
                                        game.zoom - 2, game.zoom - 2])
        
        font = pygame.font.SysFont('Calibri', 25, True, False)
        font1 = pygame.font.SysFont('Calibri', 65, True, False)
        text = font.render("Score: " + str(game.score), True, BLACK)
        text_game_over = font1.render("Game Over", True, (255, 125, 0))
        text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))

        screen.blit(text, [0, 0])
        if game.state == "gameover":
            
            screen.blit(text_game_over, [20, 200])
            screen.blit(text_game_over1, [25, 265])

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


    
if __name__ == "__main__":
    #Q_player(is_training=True)
    #Q_player(is_training=True, dqn_path="tetris_dqn_final.pth")
    Q_player(is_training=False, dqn_path="tetris_dqn_final2.pth")
    