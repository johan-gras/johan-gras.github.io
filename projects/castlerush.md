---
layout: page
title: Castle Rush
subtitle: Not Your Typical Tower Defense
bigimg: /img/projects/castlerush.png
---

*Castle Rush* is a tower defense game build in **Java**.
The *mediaval theme* is reflected ingame by both the music and the sprites.
Two modes are available, the first is a *survival mode* while the latter is a *multiplayer battle*.

![alt text](/img/projects/castlerush/menu.png "Logo Title Text 1")

## Game description

Like any tower defense game, the goal is to *defend your main base* while some enemy *try to destroy it*. Each game, a new map is generated and *special blocs of map elements* are created at random : water, lava, rock, ice. Each element having *special propriety* on the monster's movements : blocking, slowing or sliding.

Once a monster has spwawn on the map, he try to take *the shortest path* to invade your base, this mechanism is build onto the **A* algorithm**.
![alt text](/img/projects/castlerush/pathfinding.png "Monsters take the shortest path")

You can purchase *different kinds of towers or monsters* for a fixed amount of gold. Towers and monsters have *different characteristics* such as damage, range, aoe, targeting priority, special effect (slowing, ghost monster, anti-ghost,...)

![alt text](/img/projects/castlerush/types1.png "Towers")
![alt text](/img/projects/castlerush/types2.png "Monsters")
![alt text](/img/projects/castlerush/types3.png "Characteristics")

### Survival mode

This mode allow you to *defend your base* against multiple waves of *stronger and stronger openents*.
The goal is simple : *survive as long as you can*.
In order to survive, you can *buy and place tower* by purchasing and placing them.

### Multiplayer mode

This mode allow you to play versus *an other player* playing Castle Rush.
One player *host the game* and the other connect by *typing the host IP*, the connection is then made through the **TCP protocol**.
Each player start at random either by controlling *the invader* or *the defender*.
The defender can build tower like in the survival mode, while the invader can *choose his wave of monsters*.

![alt text](/img/projects/castlerush/screen.png "Logo Title Text 1")
