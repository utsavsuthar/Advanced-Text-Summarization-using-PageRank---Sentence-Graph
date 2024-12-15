#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#define MAX 100
int V,minCost=INT_MAX;
int findMinEdgeIndex(int distance[],int MST[]){
    int min=INT_MAX,index;
    for(int i=0;i<V;i++){
        if(!MST[i] && distance[i]<=min){
            min=distance[i];
            index=i;
        }
    }
    return index;
}
void findMinCost(int graph[][MAX],int currentNode, int destination, int visited[], int currentCost) {
    if (currentNode == destination) {
        if (currentCost < minCost) {
            minCost = currentCost;
        }
        return;
    }

    visited[currentNode] = 1;

    for (int v = 0; v < V; ++v) {
        if (graph[currentNode][v] != 0 && !visited[v] && currentCost + graph[currentNode][v] < minCost) {
            findMinCost(graph, v, destination, visited, currentCost + graph[currentNode][v]);
        }
    }

    visited[currentNode] = 0; 
}


int findDiffWays(int graph[][MAX],int visited[],int current,int destination,int currCost){
    if(current==destination && currCost==minCost){
        return 1;
    }
    /* if(current==destination && currCost>minCost)
        {return 0;} */
    if(currCost>minCost){
        return 0;
    }
    visited[current]=1;
    int total_path=0;
    for(int v=0;v<V;v++){
        if(graph[current][v] && !visited[v]){
            
            total_path+= findDiffWays(graph,visited,v,destination,currCost + graph[current][v]);
        }
    }
    visited[current]=0;
    return total_path;
}
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: <program name> <input_file>\n");
        return 0;
    }
    FILE *fptr = fopen("input.txt", "r");
    if (fptr == NULL)
    {
        printf("Input File not found!!");
        return 0;
    }
    int G[MAX][MAX];
    fscanf(fptr, "%d", &V);
    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V;j++){
            fscanf(fptr,"%d",&G[i][j]);
        }
    }

    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < V;j++){
            printf("%d ",G[i][j]);
        }
        printf("\n");
    }
    int distance[V];
    int MST[V];
    int parent[V];
    int visited[V];
    for (int i = 0; i < V; i++)
    {
        distance[i] = INT_MAX;
        MST[i] = 0;
        parent[i] = -1;
        visited[i]=0;
    }
    parent[0] = -1;
    distance[0] = 0;
    int source=0,destination=V-1;
    findMinCost(G,0,V-1,visited,0);
    for (int i = 0; i < V; i++)
    {
        visited[i]=0;
    }
    int diffWays=findDiffWays(G,visited,source,destination,0);
    printf("\ndifferent ways:%d & Min Cost:%d",diffWays,minCost);
}