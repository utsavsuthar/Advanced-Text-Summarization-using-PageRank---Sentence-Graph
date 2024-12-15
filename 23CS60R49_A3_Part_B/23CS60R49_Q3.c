#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#define MAX 100
int V;
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
int findMinCost(int graph[][MAX],int distance[],int MST[],int parent[]){
    int index;
    for(int i=0;i<V-1;i++){
        index=findMinEdgeIndex(distance,MST);
        MST[index]=1;
        for(int v=0;v<V;v++){
            if(graph[index][v] && !MST[v] && graph[index][v]<distance[v]){
                parent[v]=index;
                distance[v]=graph[index][v];
            }
        }
    }
    int cost=0;
    for(int i=1;i<V;i++){
        cost+=graph[parent[i]][i];
    }
    return cost;
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
    for (int i = 0; i < V; i++)
    {
        distance[i] = INT_MAX;
        MST[i] = 0;
        parent[i] = -1;
    }
    parent[0] = -1;
    distance[0] = 0;
    int cost=findMinCost(G,distance, MST, parent);
    printf("\ncost:%d",5000*cost);
}