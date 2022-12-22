// File Name: tracking.cpp
// Author: Shaoxiong Wang
// Create Time: 2018/12/20 10:11

#include "tracking_class.h"
#include <iostream>
#include <stdio.h>

double dist_sqr(Point_t a, Point_t b){
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}


double sum(Point_t a){
    return (a.x*a.x + a.y*a.y);
}

int Matching::precessor(int i, int j) {
    return (degree[i][j] <= theta 
        && degree[i][j] >= -theta 
        && Dist[i][j] <= dmax 
        && Dist[i][j] >= dmin);
}

Matching::Matching(int N_, int M_, int fps_, double x0_, double y0_, double dx_, double dy_){

    N = N_;
    M = M_;
    NM = N * M;

    fps = fps_;

    x_0 = x0_; y_0 = y0_; dx = dx_; dy = dy_;

    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            O[i][j].x = x_0 + j * dx;
            O[i][j].y = y_0 + i * dy;
        }
    }
    flag_record = 1;

    dmin = (dx * 0.5) * (dx * 0.5);
    dmax = (dx * 1.8) * (dx * 1.8);
    theta = 70;
    moving_max = dx * 2;
    flow_difference_threshold = dx * 0.8;
    cost_threshold = 15000 * (dx / 21)* (dx / 21);
}

// void Matching::init(Point_t *centers, int count){
void Matching::init(std::vector<std::vector<double>> centers) {
    int i, j;

    // read points from centers
    n = centers.size();

    for (i = 0; i < n; i++){
        C[i].x = centers[i][0];
        C[i].y = centers[i][1];
        C[i].id = i;
        // std::cout<<C[i].x<<" "<<C[i].y<<" "<<std::endl;
    }

    // init arrays for search
    memset(done, 0, sizeof(done));
    memset(occupied, -1, sizeof(occupied));
    minf = -1;

    // sort by x-axis, if same by y-axis
    std::sort(C, C+n);

    // calculate distance and angle O(N^2M^2)
    for (i = 0; i < n; i++) {
        for (j = 0; j < i; j++) {
            Dist[i][j] = dist_sqr(C[i], C[j]);
            degree[i][j] = asin(fabs(C[i].y - C[j].y) / sqrt(Dist[i][j])) * 180.0 / PI;
        }
    }
}

void Matching::run(){
    int missing, spare;

    time_st = clock();

    missing = NM - n;
    spare = n - NM;
    missing = missing < 0 ? 0 : missing;
    spare = spare < 0 ? 0 : spare;
    dfs(0, 0, missing, spare);
    for(int t=1;t<=3;t++) {
        if(minf == -1){
            // std::cout<<"TRY AGAIN!!"<<std::endl;
            memset(done, 0, sizeof(done));
            memset(occupied, -1, sizeof(occupied));
            dfs(0, 0, missing + 1, spare + 1);
        }
    }
    int i;
    if (flag_record == 1){
        flag_record = 0;
        for (i = 0; i < n; i++){
            O[MinRow[i]][MinCol[i]].x = C[i].x;
            O[MinRow[i]][MinCol[i]].y = C[i].y;
        }
    }
    // std::cout<<"MINF "<<minf<<"\t\t";
}

std::tuple<vvd, vvd, vvd, vvd, vvd> Matching::get_flow() {
    vvd Ox(N), Oy(N), Cx(N), Cy(N), Occupied(N);

    int i, j;
    for(i = 0; i < N; i++){
        Ox[i] = vd(M); Oy[i] = vd(M); Cx[i] = vd(M); Cy[i] = vd(M); Occupied[i] = vd(M);
        for(j = 0; j < M; j++){
            Ox[i][j] = O[i][j].x;
            Oy[i][j] = O[i][j].y;
            Cx[i][j] = MinD[i][j].x;
            Cy[i][j] = MinD[i][j].y;
            Occupied[i][j] = MinOccupied[i][j];
            // Point a(matcher.O[i][j].x, matcher.O[i][j].y), b(matcher.MinD[i][j].x + 2 * (matcher.MinD[i][j].x - matcher.O[i][j].x), matcher.MinD[i][j].y + 2 * (matcher.MinD[i][j].y - matcher.O[i][j].y));
        }
    }

    return std::make_tuple(Ox, Oy, Cx, Cy, Occupied);
}

double Matching::calc_cost(int i){
    double c = 0, cost = 0;
    int left, up, down;
    Point_t flow1, flow2;

    cost = cost + K1 * sum(C[i] - O[Row[i]][Col[i]]);
    flow1 = C[i] - O[Row[i]][Col[i]];

    if(Col[i] > 0){
        left = occupied[Row[i]][Col[i]-1];
        if (left > -1){
            flow2 = C[left] - O[Row[i]][Col[i]-1];
            c = sum(flow2 - flow1);
            if (sqrt(c) >= flow_difference_threshold) c = 1e8;
            cost +=  K2 * c;
        }
    }
    if(Row[i] > 0){
        up = occupied[Row[i]-1][Col[i]];
        if (up > -1){
            flow2 = C[up] - O[Row[i]-1][Col[i]];
            c = sum(flow2 - flow1);
            if (sqrt(c) >= flow_difference_threshold) c = 1e8;
            cost +=  K2 * c;
        }
    }
    if(Row[i] < N - 1){
        down = occupied[Row[i] + 1][Col[i]];
        if (down > -1){
            flow2 = C[down] - O[Row[i]+1][Col[i]];
            c = sum(flow2 - flow1);
            if (sqrt(c) >= flow_difference_threshold) c = 1e8;
            cost +=  K2 * c;
        }
    }
    return cost;
}

double Matching::infer(){
    double cost = 0;
    int boarder_nb = 0;
    int i, j, k, x, y, d=1, cnt, nx, ny, nnx, nny;

    int dir[4][2] = {{0, -1}, {-1, 0}, {0, 1}, {1, 0}};
    Point_t flow1, flow2;

    Point_t moving;

    for(i = 0; i < N; i++){
        for(j = 0;j < M; j++){
            if(occupied[i][j] <= -1){
                moving.x = 0;
                moving.y = 0;
                cnt = 0;
                for (k=0;k<4;k++){
                    nx = i + dir[k][0];
                    ny = j + dir[k][1];
                    nnx = nx + dir[k][0];
                    nny = ny + dir[k][1];
                    if (nnx < 0 || nnx >= N || nny < 0 || nny >= M) continue;
                    if (occupied[nx][ny] <= -1 || occupied[nnx][nny] <= -1) continue;
                    moving = moving + (C[occupied[nx][ny]] - O[nx][ny] + (C[occupied[nx][ny]] - O[nx][ny] - C[occupied[nnx][nny]] + O[nnx][nny]));
                    cnt += 1;
                }
                if(cnt == 0){
                    for(x=i-d;x<=i+d;x++){
                        for(y=j-d;y<=j+d;y++){
                            if (x < 0 || x >= N || y < 0 || y >= M) continue;
                            if (occupied[x][y] <= -1) continue;
                            moving = moving + (C[occupied[x][y]] - O[x][y]);
                            cnt += 1;
                        }
                    }
                }
                if(cnt == 0){
                    for(x=i-d-1;x<=i+d+1;x++){
                        for(y=j-d-1;y<=j+d+1;y++){
                            if (x < 0 || x >= N || y < 0 || y >= M) continue;
                            if (occupied[x][y] <= -1) continue;
                            moving = moving + (C[occupied[x][y]] - O[x][y]);
                            cnt += 1;
                        }
                    }
                }
                D[i][j] = O[i][j] + moving / (cnt + 1e-6);
                if (j == 0 && D[i][j].y >= O[i][j].y - dy / 2.0) boarder_nb++;
                if (j == N-1 && D[i][j].y <= O[i][j].y + dy / 2.0) boarder_nb++;
                cost = cost + K1 * sum(D[i][j] - O[i][j]);
            }
        }
    }

    if(boarder_nb >= N -1 ) cost += 1e7;

    for(i = 0; i < N; i++){
        for(j = 0;j < M; j++){
            if(occupied[i][j] <= -1){
                flow1 = D[i][j] - O[i][j];
                for (k = 0; k < 4; k++){
                    nx = i + dir[k][0];
                    ny = j + dir[k][1];
                    if (nx < 0 || nx > N - 1 || ny < 0 || ny > M -1) continue;
                    if (occupied[nx][ny] > -1){
                        flow2 = (C[occupied[nx][ny]] - O[nx][ny]);
                        cost +=  K2 * sum(flow2 - flow1);
                    }
                    else if (k < 2 && occupied[nx][ny] <= -1){
                        flow2 = (D[nx][ny] - O[nx][ny]);
                        cost +=  K2 * sum(flow2 - flow1);
                    }
                }
            }
        }
    }
    return cost;
}

void Matching::dfs(int i, double cost, int missing, int spare){
    // if(occupied[6][0] <= -1 && occupied[7][0] <= -1)
    // std::cout<<i<<" "<<"COST: "<<cost<<"fmin: "<< minf<< " missing "<<missing<<" spare "<<spare<<std::endl;
    if (((float)(clock()-time_st))/CLOCKS_PER_SEC >= 1.0 / fps) return;
    if(cost >= minf && minf != -1) return;
    if(cost >= cost_threshold) return;
    int j, k, count = 0, flag, m, same_col;
    double c;
    if (i >= n) {
        cost += infer();
        // printf("\nCOST: %lf\n", cost);
        // for (j=0;j<n;j++){
        //     printf("%d %d \t %lf %lf\n", Row[j], Col[j], C[j].x, C[j].y);
        // }
        // printf("--------------------------------------------\n");
        if (cost < minf || minf == -1) {
            // if (int(cost) == 31535) cost = 0;
            minf = cost;
            for (j=0;j<n;j++){
                // printf("%d %d \t %lf %lf\n", Row[j], Col[j], C[j].x, C[j].y);
                MinRow[j] = Row[j];
                MinCol[j] = Col[j];
                if (Row[j] < 0) continue;
                D[Row[j]][Col[j]].x = C[j].x;
                D[Row[j]][Col[j]].y = C[j].y;
            }
            for (j=0;j<N;j++){
                for (k=0;k<M;k++){
                    MinOccupied[j][k] = occupied[j][k];
                    MinD[j][k].x = D[j][k].x;
                    MinD[j][k].y = D[j][k].y;
                }
            }
        }
        return;
    }


    for (j=0;j<i;j++) {
        // if (i == 45) std::cout<<i<<" "<<j<<std::endl;

        if (precessor(i, j)) {
            Row[i] = Row[j];
            Col[i] = Col[j] + 1;
            count++;
            if (Col[i] >= M) continue;
            if (occupied[Row[i]][Col[i]] > -1) continue;
            if (Row[i] > 0 && occupied[Row[i]-1][Col[i]] > -1 && C[i].y <= C[occupied[Row[i]-1][Col[i]]].y) continue;
            if (Row[i] < N - 1 && occupied[Row[i]+1][Col[i]] > -1 && C[i].y >= C[occupied[Row[i]+1][Col[i]]].y) continue;
            int vflag = 0;
            for (k=0;k<N;k++){
                same_col = occupied[k][Col[i]];
                if(same_col > -1 && ((k < Row[i] && C[same_col].y > C[i].y) || (k > Row[i] && C[same_col].y < C[i].y))){
                    vflag = 1;
                    break;
                }
            }
            if (vflag == 1) continue;
            occupied[Row[i]][Col[i]] = i;

            c = calc_cost(i);
            dfs(i+1, cost+c, missing, spare);
            occupied[Row[i]][Col[i]] = -1;
        }
    }


    // if (count == 0) {
        for (j=0;j<N;j++) {
            if(done[j] == 0){
                flag = 0;
                for (int k = 0;k < N;k++) {
                    // printf("%d %d %d %d\t\t", k, done[k], first[k], C[i].x);
                    if (done[k] && 
                        ((k < j && first[k] > C[i].y) || (k > j && first[k] < C[i].y))
                        ){
                        flag = 1;
                        break;
                    }
                }
                if (flag == 1) continue;
                done[j] = 1;
                first[j] = C[i].y;
                Row[i] = j;
                Col[i] = 0;

                occupied[Row[i]][Col[i]] = i;
                c = calc_cost(i);

                dfs(i+1, cost+c, missing, spare);
                done[j] = 0;
                occupied[Row[i]][Col[i]] = -1;
            }
        }
    // }

    // considering missing points
    // if (C[i].y > dy && C[i].y < O[0][M-1].y - dy / 2) return;
    for(m=1;m<=missing;m++){
        for (j=0;j<N;j++) {
            // if (j >= 1 && j < N - 1) continue;
            if(fabs(C[i].y - O[j][0].y) > moving_max) continue;
            for(k=M-1;k>=0;k--) if(occupied[j][k]>-1) break;
            if(k+m+1>=M) continue;
            if (sqrt(sum(C[i] - O[j][k+m+1])) > moving_max) continue;
            for(int t=1;t<=m;t++) occupied[j][k+t] = -2;
            Row[i] = j;
            Col[i] = k + m + 1;
            c = calc_cost(i);
            occupied[Row[i]][Col[i]] = i;
            dfs(i+1, cost+c, missing - m, spare);

            for(int t=1;t<=m+1;t++) occupied[j][k+t] = -1;
        }
    }

    if (spare > 0){
        Row[i] = -1;
        Col[i] = -1;
        dfs(i+1, cost, missing, spare-1);
    }
}



std::tuple<double, double> Matching::test() {
    return std::make_tuple(dx, dy);
}

PYBIND11_MODULE(find_marker, m) {
    py::class_<Matching>(m, "Matching")
        .def(py::init<int, int, int, double, double, double, double>(), py::arg("N_") = 8, py::arg("M_") = 8, py::arg("fps_") = 30, py::arg("x0_") = 80., py::arg("y0_") = 15., py::arg("dx_") = 21., py::arg("dy_") = 21.)
        .def("run", &Matching::run)
        .def("test", &Matching::test)
        .def("init", &Matching::init)
        .def("get_flow", &Matching::get_flow);
}