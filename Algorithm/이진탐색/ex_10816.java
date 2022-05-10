import java.util.Arrays;
import java.util.Scanner;

public class ex_10816 {
    /*
    입력
    총 주어지는 숫자 카드 개수
    각 카드에 적힌 숫자
    상근이가 갖고 있는 숫자 카드 개수
    각 카드에 적힌 숫자

    출력
    상근이가 갖고 있는 숫자 카드와 동일한 숫자 카드의 개수
    */

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in); // 입력 받기 위한 세팅

        int N = sc.nextInt(); // 총 카드 개수
        int[] arr = new int[N]; // 각 카드가 갖는 숫자를 담을 배열

        for (int i = 0; i< N; i++) { // arr에 값을 하나씩 넣어 주기
            arr[i] = sc.nextInt();
        }

        Arrays.sort(arr); // 오름차순 정렬
 
        int M = sc.nextInt(); // 상근이가 갖고 있는 카드 개수

        StringBuilder sb = new StringBuilder(); // 출력을 위한 세팅

        for (int i = 0; i < M; i++) {
            int X = sc.nextInt(); // 상근이가 갖는 각 카드의 숫자
            
            sb.append(upperBound(arr, X) - lowerBound(arr, X)).append(' '); // 각 카드의 숫자 개수 찾아서 append
        }
        System.out.println(sb); // 출력
    }

    private static int lowerBound(int[] arr, int x) { // 상근이가 갖고 있는 카드의 숫자를 기준으로 배열 내부에서 가장 왼쪽 끝 인덱스 찾기
        int L = 0; // 가장 왼쪽 끝 인덱스 0으로 초기화
        int R = arr.length; // 가장 오른쪽 끝 인덱스 배열 길이로 초기화

        while(L < R) { // L이 R보다 항상 작다면, 
            int mid = ( L + R ) / 2; // L과 R의 범위의 중앙값을 mid로 정의

            if( x  <= arr[mid]) { // 상근이가 갖는 숫자가 중앙값 보다 작거나 같다면. 
                R = mid; // 가장 오른쪽 끝 값을 중앙값으로 설정
            } else { // 상근이가 갖는 숫자가 중앙값보다 크다면, 
                L = mid + 1; // 가장 왼쪽 끝 값을 중앙값보다 하나 큰 인덱스로 설정
            }
        }
        return L; // 가장 왼쪽 끝 값 반환
    }

    private static int upperBound(int[] arr, int x) { // 상근이가 갖고 있는 카드의 숫자를 기준으로 배열 내부에서 가장 왼쪽 끝 인덱스 찾기
        int L = 0; 
        int R = arr.length;

        while(L < R) {
            int mid = (L + R) / 2;
            if(x < arr[mid]) { // 상근이가 갖는 숫자가 중앙값보다 작다면,
                R = mid; // 가장 오른쪽 끝 값을 중앙값으로 설정
            } else { // 상근이가 갖는 숫자가 중앙값보다 크거나 같다면, 
                L = mid + 1; // 가장 왼쪽 끝 값을 중앙값보다 하나 큰 인덱스로 설정
            }
        }
        return L; // 가장 왼쪽 끝 값 반환
    }
}
