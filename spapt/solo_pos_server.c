#include <stdio.h>
#include <winsock2.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdbool.h>
// #include "Soloist.h"
// Main include file


// This function will print whatever the latest error was
void PrintError();

int main() {
    WSADATA wsaData;
    SOCKET listenSocket, clientSocket;
    struct sockaddr_in serverAddr, clientAddr;
    int clientAddrLen = sizeof(clientAddr);
    char buffer[1024];
    bool soloserver = true;
    struct timeval tv;
    struct tm* timeinfo;

    // SoloistHandle* handles;
    // // Handle to give access to Soloist
    // SoloistHandle handle = NULL;
    // DWORD handleCount = 0;
    // DOUBLE positionFeedback;
    // printf("Connecting to Soloist.\n");
    // if(!SoloistConnect(&handles, &handleCount)) { PrintError(); goto cleanup; }
    // if(handleCount != 1) {
    //     printf("Please make sure only one controller is configured\n");
    //     return -1;
    // }
    // handle = handles[0];
    // if(!SoloistMotionEnable(handle)) { PrintError(); goto cleanup; }

    // Initialize Winsock
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        printf("WSAStartup failed.\n");
        return 1;
    }

    // Create a listening socket
    listenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listenSocket == INVALID_SOCKET) {
        printf("Error creating socket.\n");
        WSACleanup();
        return 1;
    }

    // Bind the socket
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(7727);  // Port number
    if (bind(listenSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        printf("Bind failed.\n");
        closesocket(listenSocket);
        WSACleanup();
        return 1;
    }

    // Listen for incoming connections
    if (listen(listenSocket, SOMAXCONN) == SOCKET_ERROR) {
        printf("Listen failed.\n");
        closesocket(listenSocket);
        WSACleanup();
        return 1;
    }
    while (soloserver) {
        // // Print the value of count
        // printf("%d\n", count);
        // // Increment the counter
        // count++;
        printf("Server listening on port 7727...\n");

        // Accept a connection
        clientSocket = accept(listenSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);
        if (clientSocket == INVALID_SOCKET) {
            printf("Accept failed.\n");
            closesocket(listenSocket);
            WSACleanup();
            return 1;
        }

        printf("Client connected.\n");

        // Receive data from client
        recv(clientSocket, buffer, sizeof(buffer), 0);
        printf("Received: %s\n", buffer);


        // Get the current time
        gettimeofday(&tv, NULL);

        // Convert to local time
        timeinfo = localtime(&tv.tv_sec);

        // Extract individual components of the time
        int year = timeinfo->tm_year + 1900;  // Years since 1900
        int month = timeinfo->tm_mon + 1;      // Months since January (0-11)
        int day = timeinfo->tm_mday;           // Day of the month (1-31)
        int hour = timeinfo->tm_hour;          // Hours since midnight (0-23)
        int minute = timeinfo->tm_min;         // Minutes after the hour (0-59)
        int second = timeinfo->tm_sec;         // Seconds after the minute (0-61)
        int microsecond = tv.tv_usec;          // Microseconds (0-999999)
        
        if (strncmp(&buffer[0], "q",1)==0){
            soloserver = false;
            printf("Quit!!\n");
        }
        memset(buffer, '\0', sizeof(buffer));
        // if(!SoloistStatusGetItem(handle, STATUSITEM_PositionFeedback, &positionFeedback)) { PrintError(); goto cleanup; }
        // sprintf(buffer, "%.2f", positionFeedback);
        sprintf(buffer, "%04d-%02d-%02d %02d:%02d:%02d.%06d, %.3f,\n", year, month, day, hour, minute, second, microsecond,  13.56979);
        // sprintf(buffer, "%04d-%02d-%02d %02d:%02d:%02d.%06d, %f\n", year, month, day, hour, minute, second, microsecond,  positionFeedback);

        send(clientSocket, buffer, sizeof(buffer), 0);

        // Close sockets
        closesocket(clientSocket);
    }

    closesocket(listenSocket);
    WSACleanup();

// cleanup:
//     if(handleCount > 0) {
//         if(!SoloistMotionDisable(handle)) { PrintError(); }
//         if(!SoloistDisconnect(handles)) { PrintError(); }
//     }

// #ifdef _DEBUG
//     printf("Press ENTER to exit...\n");
//     getchar();
// #endif

    return 0;
}

// void PrintError() {
//     CHAR data[1024];
//     SoloistGetLastErrorString(data, 1024);
//     printf("Error : %s\n", data);
// }