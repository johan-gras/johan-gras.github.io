---
layout: page
title: Castle Rush
subtitle: Not Your Typical Tower Defense
bigimg: /img/projects/castlerush.png
---

[Castle Rush](https://github.com/johan-gras/Castle-Rush) is a tower defense game made in **Java** with the **LibGDX framework**.
The *medieval atmosphere* is reflected ingame by both the music and the sprites.
Two modes are available, the first is a *survival mode* and the second is a *multiplayer battle*.

![alt text](/img/projects/castlerush/menu.png "Main menu")

## Game description

Like any tower defense game, the goal is to *defend your main base* while some enemies *are trying to destroy it*. A new map is generated at every game and *special blocks of map elements* are created at random : water, lava, rock, ice. Each element triggers *a special behavior* whenever a monster walks on it : blocking, slowing or sliding.

Once a monster has spawn on the map, he tries to take *the shortest path* to invade your base.
This mechanism is built onto the **A* algorithm**.
![alt text](/img/projects/castlerush/pathfinding.png "Monsters take the shortest path")

You can purchase *different kinds of tower or monster* for a fixed amount of gold. Towers and monsters have *different characteristics* such as : damage, range, aoe, targeting priority, special effects (slowing, ghost monster, anti-ghost,...)

![alt text](/img/projects/castlerush/types1.png "Towers")
![alt text](/img/projects/castlerush/types2.png "Monsters")
![alt text](/img/projects/castlerush/types3.png "Characteristics")

### Survival mode

This mode allows you to *defend your base* against multiple waves of *stronger and stronger oppenents*.
The goal is simple : *survive as long as you can*.
In order to survive, you can *buy and place* your towers.

### Multiplayer mode

This mode allow you to play versus *an other player*.
One of the players *hosts the game* and the other connects by *typing the host IP*. 
The connection is then made through the **TCP protocol**.
Each player starts at random either by controlling *the invader* or *the defender*.
The defender can build tower like in the survival mode, while the invader can *choose his wave of monsters*.

![alt text](/img/projects/castlerush/screen.png "Game Screen")
