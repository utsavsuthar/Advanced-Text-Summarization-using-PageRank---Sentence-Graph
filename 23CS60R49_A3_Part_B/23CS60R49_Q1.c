#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#define MAX 100
void findgroup(int graph[MAX][MAX],int visited[MAX][MAX],int M,int N,int i,int j){
    visited[i][j]=1;
    //check horizontal
    if(graph[i][j+1] && !visited[i][j+1]){
        j++;
        for(int index=j;index<=N;index++){
            if(graph[i][index] && !visited[i][index]){
                visited[i][index]=1;
            }
            else
                break;
        }
        return;
    }
    if(graph[i][j-1] && !visited[i][j-1]){
        j--;
        for(int index=j;index>=1;index--){
            if(graph[i][index] && !visited[i][index]){
                visited[i][index]=1;
            }
            else
                break;
        }
        return;
    }
    //chek vertical
    if(graph[i+1][j] && !visited[i+1][j]){
        i++;
        for(int index=i;index<=M;index++){
            if(graph[index][j] && !visited[index][j]){
                visited[index][j]=1;
            }
            else
                return;
        }
        return;
    }
    if(graph[i-1][j] && !visited[i-1][j]){
        i--;
        for(int index=i;index>=1;index--){
            if(graph[index][j] && !visited[index][j]){
                visited[index][j]=1;
            }
            else
                return;
        }
        return;
    }
    
    //check diagonal
    if(graph[i+1][j+1] && !visited[i+1][j+1]){
        j++;
        i++;
        while(i<=M && j<=N){
            if(graph[i][j] && !visited[i][j]){
                visited[i][j]=1;
                i++;
                j++;
            }
            else
                return;
        }
        /* for(int index=j;index<=N;index++){
            if(graph[i][index] && !visited[i][index]){
                visited[i][index]=1;
            }
            else
                break;
        } */
        return;
    }

    if(graph[i-1][j-1] && !visited[i-1][j-1]){
        j--;
        i--;
        while(i>=1 && j>=0){
            if(graph[i][j] && !visited[i][j]){
                visited[i][j]=1;
                i--;
                j--;
            }
            else
                return;
        }
        return;
    }
    if(graph[i-1][j+1] && !visited[i-1][j+1]){
        j++;
        i--;
        while(i<=M && j<=N){
            if(graph[i][j] && !visited[i][j]){
                visited[i][j]=1;
                i--;
                j++;
            }
            else
                return;
        }
        /* for(int index=j;index<=N;index++){
            if(graph[i][index] && !visited[i][index]){
                visited[i][index]=1;
            }
            else
                break;
        } */
        return;
    }
    if(graph[i+1][j-1] && !visited[i+1][j-1]){
        j--;
        i++;
        while(i<=M && j<=N){
            if(graph[i][j] && !visited[i][j]){
                visited[i][j]=1;
                i++;
                j--;
            }
            else
                return;
        }
      
        return;
    }
}
void CountGroups(int graph[MAX][MAX],int M, int N){
    int count=0;
    int visited[MAX][MAX];
    for(int i=0;i<=M+1;i++){
        for(int j=0;j<=N+1;j++){
            visited[i][j]=0;
        }
    }

    for(int i=1;i<=M;i++){
        for(int j=1;j<=N;j++){
            if(!visited[i][j] && graph[i][j]){
                findgroup(graph,visited,M,N,i,j);
                count++;
            }
        }
    }
    printf("Total Groups are: %d",count);
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
    int M, N,G[MAX][MAX];
    fscanf(fptr, "%d %d", &M, &N);
    for (int i = 1; i <= M; i++)
    {
        for (int j = 1; j <= N;j++){
            fscanf(fptr,"%d",&G[i][j]);
        }
    }

    /* for (int i = 0; i <= M+1; i++)
    {
        for (int j = 0; j <= N+1;j++){
            printf("%d ",G[i][j]);
        }
        printf("\n");
    } */
    CountGroups(G,M,N);
}