package implement;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class BOJ15787 {
    // https://www.acmicpc.net/problem/15787
    // 이문제가 비트마스킹의 장점을 보여주긴엔 가장 좋은 문제였던 것 같다.
    // 비트마스킹의 원리랑 사용법 정도만 익히면 크게 문제없이 풀 수 있는 실버 단계의 문제이다.
    // (메모리 38736 시간 396으로 백준 Java11에서 8/79등)

    // 소요시간 >>
    // 1시간 30분 (비트마스킹 개념이랑 사용법이 미숙했다.)

    // 아이디어 >>
    // 0) n과 m을 입력받은 뒤 m줄에 걸쳐 동작코드[code], 열차번호[t], (1,2코드의 경우 승객좌석[p])을 받으며 동작한다.
    // 1) 동작에 맞춰 비트를 이동시킨다.
    //  // 1번 코드 = 열차 t의 좌석 p에 승객을 태운다 = trains[t]를 가져와 p번째 비트를 1로 변경
    //  // 2번 코드 = 열차 t의 좌석 p에 승객을 내린다 = trains[t]를 가져와 p번째 비트를 0로 변경
    //  // 3번 코드 = 열차 t의 승객을 뒤로 민다 = trains[t]를 가져와 비트를 한자리씩 <<로 밀어낸다
    //  // 4번 코드 = 열차 t의 승객을 앞으로 민다 = trains[t]를 가져와 비트를 한자리씩 >>로 밀어낸다
    // 2) Set에 trains에서 받은 값을 담아 중복 제거하고 그 수를 출력한다.
    //  // Set이 중복허용X + 순서 상관 X인 자료구조임을 이용

    // 에러로그 >>
    // 틀렸습니다 - 자리수가 20이 넘거나 1을 넘는 부분에 대한 처리를 하지 않았다.

    // 개선과정
    // * switch문으로 짰다가 후에 if문으로 바꿈.
    // => 메모리 38580 -> 38736 / 시간 396 -> 동일
    // => 같은 값으로 비교를 하는 경우 if보다 switch가 더 효율적임을 알 수 있었다.
    // * HashSet대신 Set을 사용
    // => 메모리 38580 -> 39124 / 시간 396 -> 404
    // => HashSet이 Set보다 메모리적으로도 시간적으로도 더 효율적이었다.
    public static void main(String[] args) throws IOException{
        // Input & Initialize
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st = new StringTokenizer(reader.readLine());
        int n = Integer.parseInt(st.nextToken()); // 열차의 수
        int m = Integer.parseInt(st.nextToken()); // 명령의 수
        int[] trains = new int[n]; // 열차값을 담을 배열

        // Logic
        int code, t, p=0;
        for(int i=0;i<m;i++){
            // code = 코드, t = 열차번호, p = 좌석번호
            st = new StringTokenizer(reader.readLine());
            code = Integer.parseInt(st.nextToken());
            t = Integer.parseInt(st.nextToken())-1;
            // 코드가 1,2인 경우에만 p를 받는다.
            if(code<3) p = Integer.parseInt(st.nextToken());

            switch(code){
                case 1 :
                   // trains[t] += (int)Math.pow(2,(p-1)) 와 같다.
                    trains[t] |= (1<<p-1);
                    break;
                case 2 :
                   // trains[t] -= (int)Math.pow(2,(p-1)) 와 같다.
                    trains[t] &= ~(1<<(p-1));
                    break;
                case 3 :
                    // 승객들을 뒤로 민다. (1001 -> 0100 처럼 변경)
                    trains[t] <<= 1;
                    trains[t] &= ((1<<20)-1);
                    break;
                case 4 :
                    // 승객들을 앞으로 민다. (1001 -> 0010 처럼 변경)
                    trains[t] >>= 1;
                    trains[t] &= ~0;
                    break;
            }
        }
        // Output
        HashSet<Integer> res = new HashSet<>();
        for(int val : trains) res.add(val);
        System.out.println(res.size());
    }
}
